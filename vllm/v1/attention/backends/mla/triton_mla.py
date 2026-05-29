# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
import triton.language as tl


def kvarn_mla_layout(kv_lora_rank: int, rope_dim: int, bits: int):
    """Packed-record byte layout for KVarN-MLA. Fields are 16-byte aligned so the
    decode kernel's gathered fp16 loads (scale/zp/rope, strided by REC) vectorize
    without misaligned-address faults. Returns (NB, scale_off, zp_off, rope_off, rec)."""
    NB = (kv_lora_rank * bits) // 8

    def au(x):
        return (x + 15) // 16 * 16

    scale_off = au(NB)
    zp_off = scale_off + 16
    rope_off = zp_off + 16
    rec = au(rope_off + rope_dim * 2)
    return NB, scale_off, zp_off, rope_off, rec


@triton.jit
def _kvarn_mla_decode_kernel(
    Q, Cache, BlockTable, Seqlens, O, Lse, sm_scale,
    stride_qb, stride_qh, stride_btb, stride_ob, stride_oh, stride_lb,
    L: tl.constexpr, RP: tl.constexpr, NB: tl.constexpr, REC: tl.constexpr,
    SCALE_OFF: tl.constexpr, ZP_OFF: tl.constexpr, ROPE_OFF: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Batched paged KVarN-MLA decode: grid (B, H). One (sequence, head) per
    program; tiles the sequence's tokens in BLOCK_N chunks (vectorized, not the
    old per-token serial loop), dequants the packed latent in-kernel (4-bit +
    per-token scale/zp) as K and V, adds fp16 RoPE, online softmax. cos 1.0."""
    bcur = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(Seqlens + bcur)
    offs_l = tl.arange(0, L)
    offs_p = tl.arange(0, NB)
    offs_r = tl.arange(0, RP)
    qbase = Q + bcur * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_l).to(tl.float32)
    q_rope = tl.load(qbase + L + offs_r).to(tl.float32)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([L], dtype=tl.float32)
    for start in range(0, seq_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len
        phys = tl.load(BlockTable + bcur * stride_btb + offs_n // PAGE,
                       mask=mask, other=0)
        kv_loc = phys * PAGE + offs_n % PAGE                       # [BN]
        base = kv_loc[:, None] * REC                              # [BN,1]
        b = tl.load(Cache + base + offs_p[None, :],
                    mask=mask[:, None], other=0).to(tl.uint32)    # [BN,NB]
        sc = tl.load((Cache + kv_loc * REC + SCALE_OFF).to(tl.pointer_type(tl.float16)),
                     mask=mask, other=0.0).to(tl.float32)          # [BN]
        zp = tl.load((Cache + kv_loc * REC + ZP_OFF).to(tl.pointer_type(tl.float16)),
                     mask=mask, other=0.0).to(tl.float32)
        lat_lo = (b & 0xF).to(tl.float32) * sc[:, None] + zp[:, None]
        lat_hi = ((b >> 4) & 0xF).to(tl.float32) * sc[:, None] + zp[:, None]
        lat = tl.interleave(lat_lo, lat_hi)                        # [BN,L]
        rp_ptr = (Cache + kv_loc[:, None] * REC + ROPE_OFF).to(
            tl.pointer_type(tl.float16)) + offs_r[None, :]
        rp = tl.load(rp_ptr, mask=mask[:, None], other=0.0).to(tl.float32)  # [BN,RP]
        qk = (tl.sum(lat * q_lat[None, :], axis=1)
              + tl.sum(rp * q_rope[None, :], axis=1)) * sm_scale   # [BN]
        qk = tl.where(mask, qk, -float("inf"))
        chunk_max = tl.max(qk, axis=0)
        new_max = tl.maximum(e_max, chunk_max)
        p = tl.exp(qk - new_max)                                   # [BN]
        alpha = tl.exp(e_max - new_max)
        e_sum = e_sum * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * lat, axis=0)       # [L]
        e_max = new_max
    tl.store(O + bcur * stride_ob + h * stride_oh + offs_l, (acc / e_sum).to(O.dtype.element_ty))
    tl.store(Lse + bcur * stride_lb + h, e_max + tl.log(e_sum))


@triton.jit
def _kvarn_mla_gather_dequant_kernel(
    Cache, BlockTable, TokenToSeq, CuSeqLens, SeqStarts, Dst,
    stride_bt, stride_dst,
    L: tl.constexpr, RP: tl.constexpr, NB: tl.constexpr, REC: tl.constexpr,
    SCALE_OFF: tl.constexpr, ZP_OFF: tl.constexpr, ROPE_OFF: tl.constexpr,
    PAGE: tl.constexpr,
):
    """Chunked-prefill context gather for KVarN-MLA: grid (num_tokens,). For each
    context token, maps logical position -> physical slot via the block_table,
    dequants the packed latent (4-bit + scale/zp) and copies fp16 RoPE into the
    prefill workspace [num_tokens, L+RP] in model dtype. Replaces the C++
    gather_and_maybe_dequant_cache, which assumes an fp8 cache dtype."""
    t = tl.program_id(0)
    seq = tl.load(TokenToSeq + t)
    pos = t - tl.load(CuSeqLens + seq) + tl.load(SeqStarts + seq)
    phys = tl.load(BlockTable + seq * stride_bt + pos // PAGE)
    base = (phys * PAGE + pos % PAGE) * REC
    offs_p = tl.arange(0, NB)
    offs_r = tl.arange(0, RP)
    b = tl.load(Cache + base + offs_p).to(tl.uint32)
    sc = tl.load((Cache + base + SCALE_OFF).to(tl.pointer_type(tl.float16))).to(tl.float32)
    zp = tl.load((Cache + base + ZP_OFF).to(tl.pointer_type(tl.float16))).to(tl.float32)
    lat = tl.interleave((b & 0xF).to(tl.float32) * sc + zp,
                        ((b >> 4) & 0xF).to(tl.float32) * sc + zp)
    rp = tl.load((Cache + base + ROPE_OFF).to(tl.pointer_type(tl.float16)) + offs_r).to(tl.float32)
    dbase = Dst + t * stride_dst
    tl.store(dbase + tl.arange(0, L), lat.to(Dst.dtype.element_ty))
    tl.store(dbase + L + offs_r, rp.to(Dst.dtype.element_ty))
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "kvarn_mla_k4_g128",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()
        self._is_kvarn_mla = str(self.kv_cache_dtype).startswith("kvarn_mla")
        if self._is_kvarn_mla:
            self._kvarn_bits = 4

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        """KVarN-MLA store: per-token asymmetric RTN of the latent (no Hadamard,
        to match the validated decode kernel) + fp16 RoPE, packed into 388-byte
        records and scattered at slot_mapping. Falls back to the dense MLA store
        for non-kvarn dtypes."""
        if not self._is_kvarn_mla:
            return super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if kv_cache.numel() == 0:
            return
        T = kv_c_normed.shape[0]
        R = kv_c_normed.shape[-1]
        rope = k_pe.reshape(T, -1).to(torch.float16)
        RP = rope.shape[-1]
        bits = self._kvarn_bits
        qmax = (1 << bits) - 1
        NB, SC, ZP, RO, REC = kvarn_mla_layout(R, RP, bits)
        assert REC == kv_cache.shape[-1]
        lat = kv_c_normed.float()
        lo = lat.amin(1, keepdim=True)
        hi = lat.amax(1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round((lat - lo) / scale), 0, qmax).to(torch.uint8)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()          # [T, NB]
        rec = torch.zeros(T, REC, dtype=torch.uint8, device=kv_cache.device)
        rec[:, :NB] = packed
        rec[:, SC:SC + 2] = scale.squeeze(1).to(torch.float16).view(torch.uint8).reshape(T, 2)
        rec[:, ZP:ZP + 2] = lo.squeeze(1).to(torch.float16).view(torch.uint8).reshape(T, 2)
        rec[:, RO:RO + RP * 2] = rope.reshape(T, RP).view(torch.uint8).reshape(T, RP * 2)
        kv_cache.view(-1, REC)[slot_mapping.flatten().long()] = rec

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        if self._is_kvarn_mla:
            # Fused KVarN-MLA decode: in-kernel 4-bit dequant of the packed
            # latent. kv_c_and_k_pe_cache here is the packed uint8 cache
            # [num_blocks, PAGE, REC]. q last dim = [latent(L) | rope(RP)].
            NB, SC, ZP, RO, REC = kvarn_mla_layout(
                self.kv_lora_rank, self.qk_rope_head_dim, self._kvarn_bits)
            PAGE = kv_c_and_k_pe_cache.shape[1]
            bt = attn_metadata.decode.block_table
            _kvarn_mla_decode_kernel[(B, q_num_heads)](
                q, kv_c_and_k_pe_cache, bt, attn_metadata.decode.seq_lens, o, lse,
                self.scale,
                q.stride(0), q.stride(1), bt.stride(0), o.stride(0), o.stride(1),
                lse.stride(0),
                L=self.kv_lora_rank, RP=self.qk_rope_head_dim, NB=NB, REC=REC,
                SCALE_OFF=SC, ZP_OFF=ZP, ROPE_OFF=RO, PAGE=PAGE, BLOCK_N=64,
            )
            return o, lse

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # Minimum work per split
            # hardware dependent
            min_work_per_split = 512

            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)

            # use power of 2 to avoid excessive kernel instantiations
            ideal_splits = triton.next_power_of_2(ideal_splits)

            # Calculate SM-based maximum splits with occupancy multiplier
            # 2-4x allows multiple blocks per SM for latency hiding
            # hardware dependent
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse

import importlib
import importlib.util
import math

import torch

from minilab.checks import require


_TRITON_SPEC = importlib.util.find_spec("triton")
_TRITON_LANGUAGE_SPEC = importlib.util.find_spec("triton.language") if _TRITON_SPEC is not None else None
_HAS_TRITON = _TRITON_SPEC is not None and _TRITON_LANGUAGE_SPEC is not None
_TRITON_BLOCK_M = 64
_TRITON_BLOCK_N = 64
_TRITON_NUM_WARPS = 4
_TRITON_NUM_STAGES = 3

if _HAS_TRITON:
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        Q_HEADS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SM_SCALE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        CAUSAL_OFFSET: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bq = tl.program_id(1)
        batch = pid_bq // Q_HEADS
        q_head = pid_bq - batch * Q_HEADS
        kv_head = q_head // KV_GROUP_SIZE
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        kv_head_offset = (batch * (Q_HEADS // KV_GROUP_SIZE) + kv_head) * KV_LEN * HEAD_DIM
        row_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_l = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for n_start in range(0, KV_LEN, BLOCK_N):
            k = tl.load(
                k_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(q, tl.trans(k)) * SM_SCALE
            mask = (n_start + offs_n)[None, :] < KV_LEN
            if IS_CAUSAL:
                mask = mask & ((n_start + offs_n)[None, :] <= (CAUSAL_OFFSET + offs_m)[:, None])
            scores = tl.where(mask, scores, -float("inf"))

            block_m = tl.max(scores, axis=1)
            new_m = tl.maximum(row_m, block_m)
            old_scale = tl.exp(row_m - new_m)
            old_scale = tl.where(row_m == -float("inf"), 0.0, old_scale)
            probs = tl.exp(scores - new_m[:, None])
            probs = tl.where(mask, probs, 0.0)
            row_l = row_l * old_scale + tl.sum(probs, axis=1)
            v = tl.load(
                v_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            acc = acc * old_scale[:, None] + tl.dot(probs, v)
            row_m = new_m

        valid = (offs_m < Q_LEN) & (row_l > 0.0)
        safe_l = tl.where(row_l > 0.0, row_l, 1.0)
        acc = acc / safe_l[:, None]
        tl.store(
            o_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            acc,
            mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
        )
        tl.store(
            lse_ptr + pid_bq * Q_LEN + offs_m,
            row_m + tl.log(safe_l),
            mask=valid,
        )

    @triton.jit
    def _flash_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        do_ptr,
        lse_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        Q_HEADS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SM_SCALE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        CAUSAL_OFFSET: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bq = tl.program_id(1)
        batch = pid_bq // Q_HEADS
        q_head = pid_bq - batch * Q_HEADS
        kv_head = q_head // KV_GROUP_SIZE
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        out = tl.load(
            o_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            lse_ptr + pid_bq * Q_LEN + offs_m,
            mask=offs_m < Q_LEN,
            other=float("inf"),
        )
        kv_head_offset = (batch * (Q_HEADS // KV_GROUP_SIZE) + kv_head) * KV_LEN * HEAD_DIM
        delta = tl.sum(out * do, axis=1)
        dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for n_start in range(0, KV_LEN, BLOCK_N):
            k = tl.load(
                k_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(q, tl.trans(k)) * SM_SCALE
            mask = (n_start + offs_n)[None, :] < KV_LEN
            if IS_CAUSAL:
                mask = mask & ((n_start + offs_n)[None, :] <= (CAUSAL_OFFSET + offs_m)[:, None])
            p = tl.exp(scores - lse[:, None])
            p = tl.where(mask, p, 0.0)

            dv = tl.dot(tl.trans(p), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * SM_SCALE
            dq += tl.dot(ds, k)
            dk = tl.dot(tl.trans(ds), q)

            tl.atomic_add(
                dv_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                dv,
                sem="relaxed",
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
            )
            tl.atomic_add(
                dk_ptr + kv_head_offset + (n_start + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                dk,
                sem="relaxed",
                mask=((n_start + offs_n)[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM),
            )

        tl.store(
            dq_ptr + pid_bq * Q_LEN * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM),
        )
else:
    triton = None


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def _validate_triton_flash_inputs(q, k, v, attn_bias, dropout_p, is_causal, causal_offset, kv_group_size):
    require(_HAS_TRITON, "backend='flash' requires the triton package")
    require(type(causal_offset) is int, "backend='flash' causal_offset must be an integer")
    require(type(kv_group_size) is int, "backend='flash' kv_group_size must be an integer")
    require(q.is_cuda and k.is_cuda and v.is_cuda, "backend='flash' requires CUDA tensors")
    require(q.dtype in {torch.float16, torch.bfloat16}, "backend='flash' requires fp16 or bf16 inputs")
    require(k.dtype == q.dtype and v.dtype == q.dtype, "backend='flash' requires matching q/k/v dtypes")
    require(attn_bias is None, "backend='flash' currently supports only dense causal/non-causal attention")
    require(dropout_p == 0.0, "backend='flash' does not support attention dropout yet")
    require(q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "backend='flash' expects (B, H, T, D)")
    require(q.size(-2) > 0 and k.size(-2) > 0, "backend='flash' requires non-empty q/k sequences")
    require(q.size(-1) > 0, "backend='flash' requires head_dim > 0")
    require(q.size(0) == k.size(0) == v.size(0), "backend='flash' requires matching batch dims")
    require(k.size(1) == v.size(1), "backend='flash' requires matching K/V head dims")
    require(kv_group_size > 0, "backend='flash' kv_group_size must be > 0")
    require(q.size(1) == k.size(1) * kv_group_size, "backend='flash' query heads must equal key heads * kv_group_size")
    require(k.size(-2) == v.size(-2), "backend='flash' requires matching K/V sequence lengths")
    require(q.size(-1) == k.size(-1) == v.size(-1), "backend='flash' requires matching head dims")
    require(q.size(-1) <= 128, "backend='flash' supports head_dim <= 128")
    require(_next_power_of_2(q.size(-1)) <= 128, "backend='flash' supports power-of-two block dim <= 128")
    require(causal_offset >= 0, "backend='flash' causal_offset must be >= 0")
    require(causal_offset == 0 or is_causal, "backend='flash' causal_offset only applies to causal attention")
    require(not is_causal or q.size(-2) + causal_offset == k.size(-2), (
        "backend='flash' causal mode expects kv_len == q_len + causal_offset"
    ))


class _TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, causal_offset, kv_group_size):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        B, q_heads, q_len, head_dim = q.shape
        kv_len = k.size(-2)
        block_d = _next_power_of_2(head_dim)
        sm_scale = 1.0 / math.sqrt(head_dim)
        out = torch.empty_like(q)
        lse = torch.empty(B, q_heads, q_len, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(q_len, _TRITON_BLOCK_M), B * q_heads)
        _flash_fwd_kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            q_len,
            kv_len,
            q_heads,
            kv_group_size,
            head_dim,
            sm_scale,
            is_causal,
            causal_offset,
            BLOCK_M=_TRITON_BLOCK_M,
            BLOCK_N=_TRITON_BLOCK_N,
            BLOCK_D=block_d,
            num_warps=_TRITON_NUM_WARPS,
            num_stages=_TRITON_NUM_STAGES,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        ctx.causal_offset = causal_offset
        ctx.kv_group_size = kv_group_size
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        do = do.contiguous()
        B, q_heads, q_len, head_dim = q.shape
        kv_len = k.size(-2)
        block_d = _next_power_of_2(head_dim)
        sm_scale = 1.0 / math.sqrt(head_dim)
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        grid = (triton.cdiv(q_len, _TRITON_BLOCK_M), B * q_heads)
        _flash_bwd_kernel[grid](
            q,
            k,
            v,
            out,
            do,
            lse,
            dq,
            dk,
            dv,
            q_len,
            kv_len,
            q_heads,
            ctx.kv_group_size,
            head_dim,
            sm_scale,
            ctx.is_causal,
            ctx.causal_offset,
            BLOCK_M=_TRITON_BLOCK_M,
            BLOCK_N=_TRITON_BLOCK_N,
            BLOCK_D=block_d,
            num_warps=_TRITON_NUM_WARPS,
            num_stages=_TRITON_NUM_STAGES,
        )
        return dq, dk, dv, None, None, None


def triton_flash_attention(q, k, v, attn_bias=None, dropout_p=0.0, is_causal=False, causal_offset=0, kv_group_size=1):
    _validate_triton_flash_inputs(q, k, v, attn_bias, dropout_p, is_causal, causal_offset, kv_group_size)
    return _TritonFlashAttention.apply(q, k, v, bool(is_causal), causal_offset, kv_group_size)

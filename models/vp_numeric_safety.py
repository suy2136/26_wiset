"""Retry-only FP16 numerical safety for VP compact NBS inference.

Healthy VP calls keep Transformers' original attention implementation and the
compact LoRA fast path.  If the final VP prediction contains NaN/Inf, the same
call is retried with pre-scaled Q/K attention and detailed compact-LoRA range
handling.  A final FP32-score retry is retained as a last-resort diagnostic.
"""

from contextlib import contextmanager
import math
from types import MethodType

import torch
import torch.nn.functional as F


def _prescaled_qk_scores(query_states, key_states, head_dim):
    operand_scale = math.sqrt(math.sqrt(head_dim))
    return torch.matmul(
        query_states / operand_scale,
        (key_states / operand_scale).transpose(2, 3),
    )


def _safe_llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    padding_mask=None,
):
    """Transformers 4.34 Llama attention with safe score formation."""
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        kv_slice = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0,
        )
        key_slices = self.k_proj.weight.split(kv_slice, dim=0)
        value_slices = self.v_proj.weight.split(kv_slice, dim=0)
        query_states = torch.cat([
            F.linear(hidden_states, query_slices[index])
            for index in range(self.config.pretraining_tp)
        ], dim=-1)
        key_states = torch.cat([
            F.linear(hidden_states, key_slices[index])
            for index in range(self.config.pretraining_tp)
        ], dim=-1)
        value_states = torch.cat([
            F.linear(hidden_states, value_slices[index])
            for index in range(self.config.pretraining_tp)
        ], dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    next_past_key_value = (key_states, value_states) if use_cache else None
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    mode = getattr(self, "_vp_safe_attention_mode", "fp16_prescaled")
    if mode == "fp16_prescaled":
        attn_weights = _prescaled_qk_scores(
            query_states, key_states, self.head_dim
        )
    elif mode == "fp32":
        attn_weights = torch.matmul(
            query_states.float(), key_states.transpose(2, 3).float()
        ) / math.sqrt(self.head_dim)
    else:
        raise ValueError(f"unsupported VP safe attention mode: {mode}")

    expected_weights = (bsz, self.num_heads, q_len, kv_seq_len)
    if attn_weights.size() != expected_weights:
        raise ValueError(
            f"attention weights should be {expected_weights}, got "
            f"{tuple(attn_weights.size())}"
        )
    if attention_mask is not None:
        expected_mask = (bsz, 1, q_len, kv_seq_len)
        if attention_mask.size() != expected_mask:
            raise ValueError(
                f"attention mask should be {expected_mask}, got "
                f"{tuple(attention_mask.size())}"
            )
        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
    expected_output = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != expected_output:
        raise ValueError(
            f"attention output should be {expected_output}, got "
            f"{tuple(attn_output.size())}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        bsz, q_len, self.hidden_size
    )
    if self.config.pretraining_tp > 1:
        chunks = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        output_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            F.linear(chunks[index], output_slices[index])
            for index in range(self.config.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)
    return (
        attn_output,
        attn_weights if output_attentions else None,
        next_past_key_value,
    )


@contextmanager
def vp_safe_retry(model, mode):
    """Temporarily enable detailed compact-LoRA and safe Q/K computation."""
    missing = object()
    attention_state = []
    compact_modules = []
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            previous = module.__dict__.get("forward", missing)
            attention_state.append((module, previous))
            module.forward = MethodType(_safe_llama_attention_forward, module)
            module._vp_safe_attention_mode = mode
        if hasattr(module, "_vp_detailed_safety_active"):
            compact_modules.append(module)
            module._vp_detailed_safety_active = True
    try:
        yield
    finally:
        for module in compact_modules:
            module._vp_detailed_safety_active = False
        for module, previous in attention_state:
            module.__dict__.pop("_vp_safe_attention_mode", None)
            if previous is missing:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous


def enable_vp_fp16_fallback(model):
    """Enable retry behavior only on VP networking-head Llama instances."""
    enabled = 0
    for module in model.modules():
        if module.__class__.__name__ != "LlamaNetworkingHeadModel":
            continue
        module._vp_fp16_fallback_enabled = True
        module.vp_fp16_fallback_calls = int(
            getattr(module, "vp_fp16_fallback_calls", 0)
        )
        module.vp_fp32_fallback_calls = int(
            getattr(module, "vp_fp32_fallback_calls", 0)
        )
        enabled += 1
    return enabled


def vp_numeric_safety_report(model):
    targets = [
        module for module in model.modules()
        if module.__class__.__name__ == "LlamaNetworkingHeadModel"
        and getattr(module, "_vp_fp16_fallback_enabled", False)
    ]
    return {
        "enabled_models": len(targets),
        "fp16_prescaled_qk_fallback_calls": sum(
            int(getattr(module, "vp_fp16_fallback_calls", 0))
            for module in targets
        ),
        "fp32_attention_fallback_calls": sum(
            int(getattr(module, "vp_fp32_fallback_calls", 0))
            for module in targets
        ),
    }

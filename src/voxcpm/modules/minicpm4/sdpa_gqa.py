import math
import torch


def sdpa_gqa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_mask: torch.Tensor = None,
    is_causal: bool = False,
    scaling_factor: float = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention with grouped-query attention (GQA),
    implemented with primitive ops for ONNX export.

    This function matches torch.nn.functional.scaled_dot_product_attention
    semantics when called with enable_gqa=True:
    - Default scaling is 1/sqrt(D); custom scaling_factor multiplies the scores.
    - is_causal applies an upper-triangular allow mask (j <= i) per (Q,K).
    - attn_mask can be:
      * boolean: True means ALLOW, broadcastable to (B, H_q, Q, K)
      * numeric: additive bias added to the attention scores, broadcastable
    - KV heads are expanded to match query heads via grouping (GQA).

    Shapes:
      query_states: (B, H_q, Q, D)
      key_states:   (B, H_kv, K, D)
      value_states: (B, H_kv, K, D)
    Returns:
      attn_output:  (B, H_q, Q, D)
    """
    b, h_q, q_len, d = query_states.shape
    b2, h_kv, k_len, d2 = key_states.shape
    assert b == b2 and d == d2, "Incompatible shapes for attention"

    # make contiguous for accelerator compatibility
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    # expand KV heads to match query heads via GQA grouping
    assert h_q % h_kv == 0, "num_heads must be divisible by num_kv_heads for GQA"
    groups = h_q // h_kv

    key_exp = key_states.unsqueeze(2).expand(b, h_kv, groups, k_len, d).reshape(b, h_q, k_len, d)
    value_exp = value_states.unsqueeze(2).expand(b, h_kv, groups, k_len, d).reshape(b, h_q, k_len, d)

    # attention scores with flexible scaling (match torch SDPA default)
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(d)
    scores = torch.matmul(query_states, key_exp.transpose(-2, -1)) * scaling_factor  # (b, h_q, q_len, k_len)

    # boolean allow mask and additive bias (numeric)
    allow_mask = None  # boolean mask: True means ALLOW
    additive_bias = None  # numeric mask: added to scores

    # causal masking: allow j <= i
    if is_causal:
        causal_allow = torch.ones((q_len, k_len), dtype=torch.bool, device=query_states.device).tril()
        allow_mask = causal_allow.view(1, 1, q_len, k_len).expand(b, h_q, q_len, k_len)

    # attn_mask processing
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            m = attn_mask
            if m.dim() == 1:
                m = m.view(1, 1, 1, k_len).expand(b, h_q, q_len, k_len)
            elif m.dim() == 2:
                m = m.view(1, 1, q_len, k_len).expand(b, h_q, q_len, k_len)
            elif m.dim() == 4:
                if m.shape != (b, h_q, q_len, k_len):
                    m = m.expand(b, h_q, q_len, k_len)
            else:
                m = m.view(1, 1, 1, 1).expand(b, h_q, q_len, k_len)
            allow_mask = m if allow_mask is None else (allow_mask & m)
        else:
            # treat non-bool mask as additive bias
            m = attn_mask
            if m.dim() == 1:
                m = m.view(1, 1, 1, k_len).expand(b, h_q, q_len, k_len)
            elif m.dim() == 2:
                m = m.view(1, 1, q_len, k_len).expand(b, h_q, q_len, k_len)
            elif m.dim() == 4:
                if m.shape != (b, h_q, q_len, k_len):
                    m = m.expand(b, h_q, q_len, k_len)
            else:
                m = m.view(1, 1, 1, 1).expand(b, h_q, q_len, k_len)
            additive_bias = m

    if additive_bias is not None:
        scores = scores + additive_bias

    # masked softmax: ensure rows fully masked produce all-zero weights
    if allow_mask is None:
        attn_weights = torch.softmax(scores, dim=-1)
    else:
        # compute per-row max over allowed positions; -inf for disallowed
        neg_inf = torch.tensor(float('-inf'), dtype=scores.dtype, device=scores.device)
        scores_allowed = torch.where(allow_mask, scores, neg_inf)
        max_scores = scores_allowed.max(dim=-1).values  # (b, h_q, q_len)
        # subtract max for stability and exponentiate only where allowed
        exps = torch.exp(torch.where(allow_mask, scores - max_scores[..., None], neg_inf))
        exps = torch.where(allow_mask, exps, torch.zeros_like(exps))
        sums = exps.sum(dim=-1, keepdim=True)
        attn_weights = torch.where(sums > 0, exps / sums, torch.zeros_like(exps))

    attn_output = torch.matmul(attn_weights, value_exp)  # (b, h_q, q_len, d)
    return attn_output

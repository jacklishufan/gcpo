import torch,math,re
import torch.nn.functional as F
@torch.no_grad()
def compute_importance_weights(answer_token_log_probs, negative_answer_token_log_probs, response_mask=None, importance_weighting_type="abs",importance_normalization='softmax', importance_normalization_temperature=1.0):
    # answer_token_log_probs: (bs, response_length) - log prob of true response token
    # negative_answer_token_log_probs: (bs, response_length) - log prob of true response token under negative prompt
    # response_mask: (bs, response_length) - mask for valid response tokens (1 for valid, 0 for padding)

    if importance_weighting_type == "kl":
        # Pointwise KL using stable low-variance estimator (Schulman 2020)
        # KL(q || p) = E_q[log q - log p] ≈ e^(log p - log q) - (log p - log q) - 1
        kl_diff = (negative_answer_token_log_probs - answer_token_log_probs).clamp(-20.0, 20.0)
        divergences = (kl_diff.exp() - kl_diff - 1).contiguous()
        divergences = torch.clamp(divergences, min=-10.0, max=10.0)
    else:
        raise NotImplementedError(f"Unknown importance_weighting_type: {importance_weighting_type}")

    # normalize
    if importance_normalization.startswith("hist"):
        x = 40 / 100.0  # top fraction (e.g. 0.20)
        y = 80 / 100.0  # target mass  (e.g. 0.80)
        alpha = math.log(1.0 - y) / math.log(1.0 - x) - 1.0
        weights = torch.zeros_like(divergences)
        bs = divergences.size(0)
        for i in range(bs):
            valid_mask = response_mask[i].bool() if response_mask is not None else torch.ones(divergences.size(1), dtype=torch.bool, device=divergences.device)
            valid_divs = divergences[i][valid_mask]
            N = valid_divs.size(0)
            if N == 0:
                continue
            # Rank lowest=1, highest=N; convert to percentile in (0, 1]
            order = valid_divs.argsort()
            ranks = torch.empty(N, dtype=divergences.dtype, device=divergences.device)
            ranks[order] = torch.arange(1, N + 1, dtype=divergences.dtype, device=divergences.device)
            percentiles = ranks / N  # in (1/N, 1]
            # Power-law transform; renormalize to sum = N
            w = percentiles.pow(alpha)
            w = w / w.sum() * N
            weights[i][valid_mask] = w
    else:
        weights = divergences

    return weights
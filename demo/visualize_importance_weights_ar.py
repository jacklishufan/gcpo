import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def maybe_set_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def get_prefix_token_id(tokenizer) -> int:
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token_id is not None:
            return token_id
    raise ValueError("Tokenizer must define at least one of bos_token_id, eos_token_id, or pad_token_id.")


def tokenize_prompt(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)


def tokenize_answer(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_baseline_prompt_ids(tokenizer, negative_prompt: str) -> list[int]:
    if negative_prompt:
        return tokenize_prompt(tokenizer, negative_prompt)
    return [get_prefix_token_id(tokenizer)]


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_prompt_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    prompt_ids = tokenize_prompt(tokenizer, question)[:max_prompt_len]
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def compute_answer_distributions(
    model,
    answer_ids: list[int],
    context_ids: list[int],
    device: torch.device,
):
    if not answer_ids:
        raise ValueError("Answer must contain at least one token.")

    input_ids_list = context_ids + answer_ids
    input_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits, dim=-1)
    answer_start = len(context_ids)
    answer_positions = torch.arange(answer_start - 1, answer_start - 1 + len(answer_ids), device=device)
    answer_token_ids = torch.tensor(answer_ids, device=device, dtype=torch.long)
    answer_log_probs = log_probs[0, answer_positions]
    answer_token_log_probs = answer_log_probs.gather(dim=-1, index=answer_token_ids.unsqueeze(-1)).squeeze(-1)
    return answer_ids, answer_log_probs, answer_token_log_probs


def normalize_divergence(
    divergence: torch.Tensor,
    divergence_metric: str,
    use_softmax_norm: bool,
    normalization_temperature: float,
) -> torch.Tensor:
    # if divergence_metric == "information_gain":
    #     return divergence
    if use_softmax_norm:
        scale = divergence.max().clamp(min=1e-8)
        return F.softmax(divergence / scale / normalization_temperature, dim=0)
    max_div = divergence.max().clamp(min=1e-8)
    min_div = divergence.min()
    return (divergence - min_div) / (max_div - min_div + 1e-8)


def compute_all_divergences(
    answer_log_probs: torch.Tensor,
    negative_answer_log_probs: torch.Tensor,
    answer_token_log_probs: torch.Tensor,
    negative_answer_token_log_probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probs = torch.exp(answer_log_probs)
    probs_negative = torch.exp(negative_answer_log_probs)
    #kl_forward = (probs * (answer_log_probs - negative_answer_log_probs)).sum(dim=-1)
    #kl_backward = (probs_negative * (negative_answer_log_probs - answer_log_probs)).sum(dim=-1)
    log_p_diff_stable = (negative_answer_log_probs - answer_log_probs).clamp(-20.0, 20.0)
    low_var_forward_kl = (log_p_diff_stable.exp() - log_p_diff_stable - 1).contiguous().clamp(min=0.0, max=10.0)
    low_var_reverse_kl = ((-log_p_diff_stable).exp() + log_p_diff_stable - 1).contiguous().clamp(min=0.0, max=10.0)
    js_div = (low_var_forward_kl + low_var_reverse_kl) / 2.0
    information_gain = (answer_token_log_probs - negative_answer_token_log_probs)
    return {
        "abs": torch.abs(answer_token_log_probs - negative_answer_token_log_probs),
        "kl":low_var_forward_kl,
        "js": js_div,
        "information_gain": information_gain.clamp(min=0,max=information_gain.float().quantile(0.9)),
    }


def compute_importance_weights(
    model,
    tokenizer,
    question: str,
    answer: str,
    negative_prompt: str,
    max_prompt_len: int,
    max_gen_length: int,
    use_softmax_norm: bool,
    normalization_temperature: float,
    device: torch.device,
):
    """
    Compute importance weights by comparing model outputs on:
    - prompt|answer
    - negative_prompt|answer

    This measures how much each answer token depends on the original prompt relative to
    a baseline prompt, which can be empty/pad-like or an explicit negative prompt.
    """
    max_total_len = max_prompt_len + max_gen_length
    question_ids = tokenize_prompt(tokenizer, question)[:max_prompt_len]
    negative_prompt_ids = build_baseline_prompt_ids(tokenizer, negative_prompt)[:max_prompt_len]
    answer_ids = tokenize_answer(tokenizer, answer)

    max_answer_len = min(
        len(answer_ids),
        max_total_len - len(question_ids),
        max_total_len - len(negative_prompt_ids),
    )
    if max_answer_len <= 0:
        raise ValueError("Prompt or negative prompt is too long; no room remains for answer tokens.")
    answer_ids = answer_ids[:max_answer_len]

    answer_ids, answer_log_probs, answer_token_log_probs = compute_answer_distributions(
        model=model,
        answer_ids=answer_ids,
        context_ids=question_ids,
        device=device,
    )
    negative_answer_ids, negative_answer_log_probs, negative_answer_token_log_probs = compute_answer_distributions(
        model=model,
        answer_ids=answer_ids,
        context_ids=negative_prompt_ids,
        device=device,
    )

    tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    divergences = compute_all_divergences(
        answer_log_probs=answer_log_probs,
        negative_answer_log_probs=negative_answer_log_probs,
        answer_token_log_probs=answer_token_log_probs,
        negative_answer_token_log_probs=negative_answer_token_log_probs,
    )
    weights_by_metric = {
        metric: normalize_divergence(values, metric, use_softmax_norm, normalization_temperature).detach().cpu().tolist()
        for metric, values in divergences.items()
    }
    return tokens, weights_by_metric


def plot_importance(tokens, weights, output_png: Path, top_k: int):
    """Plot importance weights as horizontal bar chart."""
    pairs = list(zip(tokens, weights))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    if top_k > 0:
        pairs_to_plot = pairs_sorted[:top_k]
    else:
        pairs_to_plot = pairs

    labels = [t.replace("\n", "\\n") for t, _ in pairs_to_plot]
    vals = [w for _, w in pairs_to_plot]

    height = max(4, 0.35 * len(vals))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("importance weight")
    ax.set_title("Token-level Importance Weights (AR Model)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def token_to_display(token: str) -> str:
    """Improve readability for common tokenizer space markers."""
    if token.startswith("\u0120"):
        return " " + token[1:]
    if token.startswith("\u2581"):
        return " " + token[1:]
    return token


def weight_to_rgb(weight: float, w_min: float, w_max: float):
    """Convert weight to RGB color (light blue -> warm red gradient)."""
    if w_max <= w_min:
        x = 0.0
    else:
        x = (weight - w_min) / (w_max - w_min)
    # Light blue -> warm red gradient
    r0, g0, b0 = (230, 243, 255)
    r1, g1, b1 = (220, 53, 69)
    r = int(r0 + (r1 - r0) * x)
    g = int(g0 + (g1 - g0) * x)
    b = int(b0 + (b1 - b0) * x)
    return r, g, b


def render_html_heatmap(
    tokens,
    weights_by_metric,
    selected_metric: str,
    question: str,
    answer: str,
    negative_prompt: str,
    output_html: Path,
):
    """Render importance weights as interactive HTML heatmap with rounded token pills."""
    if len(tokens) == 0:
        raise ValueError("No response tokens to visualize.")

    metric_options = ["abs", "kl", "information_gain"]
    tokens_display = [token_to_display(token) for token in tokens]
    option_html = "".join(
        f'<option value="{metric}"' + (" selected" if metric == selected_metric else "") + f'>{metric}</option>'
        for metric in metric_options
    )
    payload = {
        "tokens": tokens,
        "displayTokens": tokens_display,
        "weightsByMetric": weights_by_metric,
        "selectedMetric": selected_metric,
    }

    html_text = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Importance Heatmap (AR Model)</title>
    <style>
        body {{
            margin: 24px;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
            color: #222;
            background: #fafafa;
        }}
        .card {{
            background: #fff;
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .label {{
            font-size: 13px;
            color: #666;
            margin-bottom: 6px;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .select {{
            font: inherit;
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid #d0d0d0;
            background: #fff;
        }}
        .text {{
            white-space: pre-wrap;
            line-height: 1.5;
        }}
        .tokens {{
            line-height: 2.0;
        }}
        .tok {{
            display: inline-block;
            margin: 2px 3px 2px 0;
            padding: 3px 7px;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,0.08);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 12px;
        }}
        .legend {{
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgb(230,243,255), rgb(220,53,69));
            border: 1px solid #ddd;
            margin-top: 8px;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class=\"card\">
        <div class=\"label\">Question</div>
        <div class=\"text\">{html.escape(question)}</div>
    </div>

    <div class="card">
        <div class="label">Negative Prompt</div>
        <div class="text">{html.escape(negative_prompt) if negative_prompt else "&lt;empty baseline&gt;"}</div>
    </div>

    <div class=\"card\">
        <div class=\"label\">Answer</div>
        <div class=\"text\">{html.escape(answer)}</div>
    </div>

    <div class=\"card\">
        <div class=\"label\">Token Importance Heatmap (AR Model)</div>
        <div class="controls">
            <label for="metric-select">Divergence metric</label>
            <select id="metric-select" class="select">{option_html}</select>
        </div>
        <div class="tokens" id="tokens"></div>
        <div class=\"legend\"></div>
        <div class="legend-labels"><span id="legend-low"></span><span id="legend-high"></span></div>
    </div>

    <script>
        const payload = {json.dumps(payload)};

        function weightToRgb(weight, wMin, wMax) {{
            let x = 0.0;
            if (wMax > wMin) {{
                x = (weight - wMin) / (wMax - wMin);
            }}
            const r0 = 230;
            const g0 = 243;
            const b0 = 255;
            const r1 = 220;
            const g1 = 53;
            const b1 = 69;
            const r = Math.round(r0 + (r1 - r0) * x);
            const g = Math.round(g0 + (g1 - g0) * x);
            const b = Math.round(b0 + (b1 - b0) * x);
            return [r, g, b];
        }}

        function escapeHtml(text) {{
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }}

        function renderMetric(metric) {{
            const weights = payload.weightsByMetric[metric];
            const wMin = Math.min(...weights);
            const wMax = Math.max(...weights);
            const container = document.getElementById('tokens');
            const spans = payload.tokens.map((token, index) => {{
                const display = escapeHtml(payload.displayTokens[index]);
                const rawToken = escapeHtml(token);
                const weight = weights[index];
                const [r, g, b] = weightToRgb(weight, wMin, wMax);
                const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
                const textColor = luminance > 160 ? '#111' : '#fff';
                return `<span class="tok" style="background: rgb(${{r}},${{g}},${{b}}); color: ${{textColor}};" title="token=${{rawToken}} | metric=${{metric}} | weight=${{weight.toFixed(6)}}">${{display}}</span>`;
            }});
            container.innerHTML = spans.join('');
            document.getElementById('legend-low').textContent = `low (${{wMin.toFixed(4)}})`;
            document.getElementById('legend-high').textContent = `high (${{wMax.toFixed(4)}})`;
        }}

        const metricSelect = document.getElementById('metric-select');
        metricSelect.addEventListener('change', (event) => renderMetric(event.target.value));
        renderMetric(payload.selectedMetric);
    </script>
</body>
</html>
"""

    output_html.write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize importance weights for AR model (question/answer pair).")
    parser.add_argument("--model", required=True, help="Path or HF id for the AR model checkpoint")
    parser.add_argument("--question", required=True, help="Input question/prompt")
    parser.add_argument("--answer", default=None, help="Reference answer/response")
    parser.add_argument("--negative-prompt", default="", help="Baseline prompt used for comparison")
    parser.add_argument("--use-generated-answer", action="store_true", help="Generate the answer from the model instead of using --answer")

    parser.add_argument("--output-dir", default="importance_vis_ar", help="Where to write output files")
    parser.add_argument("--output-name", default="importance", help="Base output filename")

    parser.add_argument("--max-prompt-len", type=int, default=8000)
    parser.add_argument("--max-gen-length", type=int, default=377)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-do-sample", action="store_true")
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)

    parser.add_argument("--divergence-metric", choices=["abs", "kl", "information_gain"], default="abs")
    parser.add_argument("--importance-softmax-normalization", action="store_true")
    parser.add_argument("--importance-normalization-temperature", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--top-k", type=int, default=0, help="Plot top-k tokens only; 0 plots all")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    maybe_set_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype="auto")
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.eval()

    if args.use_generated_answer:
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.generation_max_new_tokens,
            do_sample=args.generation_do_sample,
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            device=device,
        )
    elif args.answer is not None:
        answer = args.answer
    else:
        raise ValueError("Provide --answer or set --use-generated-answer.")

    tokens, weights_by_metric = compute_importance_weights(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        answer=answer,
        negative_prompt=args.negative_prompt,
        max_prompt_len=args.max_prompt_len,
        max_gen_length=args.max_gen_length,
        use_softmax_norm=args.importance_softmax_normalization,
        normalization_temperature=args.importance_normalization_temperature,
        device=device,
    )
    weights = weights_by_metric[args.divergence_metric]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{args.output_name}.png"
    txt_path = out_dir / f"{args.output_name}.txt"
    html_path = out_dir / f"{args.output_name}.html"

    plot_importance(tokens, weights, png_path, args.top_k)
    render_html_heatmap(
        tokens,
        weights_by_metric,
        args.divergence_metric,
        args.question,
        answer,
        args.negative_prompt,
        html_path,
    )

    with txt_path.open("w", encoding="utf-8") as f:
        for t, w in zip(tokens, weights):
            f.write(f"{t}\t{w:.6f}\n")

    print(f"Saved plot: {png_path}")
    print(f"Saved weights: {txt_path}")
    print(f"Saved html heatmap: {html_path}")
    if args.use_generated_answer:
        print(f"Generated answer: {answer}")


if __name__ == "__main__":
    main()

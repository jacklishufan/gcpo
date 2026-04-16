# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Visualization utilities for importance weights in RL training.
"""
import html
from typing import Any

import torch

try:
    import wandb
except ImportError:
    wandb = None


def token_to_display(token: str) -> str:
    """Improve readability for common tokenizer space markers."""
    if token.startswith("\u0120"):
        return " " + token[1:]
    if token.startswith("\u2581"):
        return " " + token[1:]
    return token


def weight_to_rgb(weight: float, w_min: float, w_max: float) -> tuple[int, int, int]:
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
    tokens: list[str],
    weights: list[float],
    question: str,
    answer: str,
    negative_prompt: str,
    metric_name: str = "importance",
) -> str:
    """Render importance weights as HTML heatmap with token pills.
    
    Args:
        tokens: List of token strings
        weights: List of importance weight values for each token
        question: The original question/prompt
        answer: The model's generated answer
        negative_prompt: The negative prompt baseline
        metric_name: Name of the weighting metric (e.g., 'abs', 'kl', 'information_gain')
    
    Returns:
        HTML string with interactive visualization
    """
    if len(tokens) == 0:
        return "<p>No tokens to visualize.</p>"

    tokens_display = [token_to_display(token) for token in tokens]
    weights_tensor = torch.tensor(weights)
    w_min = weights_tensor.min().item()
    w_max = weights_tensor.max().item()

    token_html = ""
    for token, display_token, weight in zip(tokens, tokens_display, weights):
        r, g, b = weight_to_rgb(weight, w_min, w_max)
        token_html += f'<span class="tok" style="background-color: rgb({r},{g},{b});" title="{weight:.4f}">{html.escape(display_token)}</span>'

    html_text = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Importance Weights Heatmap</title>
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
            font-weight: 600;
        }}
        .text {{
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 14px;
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

    <div class=\"card\">
        <div class=\"label\">Negative Prompt</div>
        <div class=\"text\">{html.escape(negative_prompt) if negative_prompt else "&lt;empty baseline&gt;"}</div>
    </div>

    <div class=\"card\">
        <div class=\"label\">Answer</div>
        <div class=\"text\">{html.escape(answer)}</div>
    </div>

    <div class=\"card\">
        <div class=\"label\">Token {metric_name.capitalize()} Weights</div>
        <div class=\"tokens\">{token_html}</div>
        <div class=\"legend\"></div>
        <div class=\"legend-labels\">
            <span>Low</span>
            <span>High</span>
        </div>
    </div>
</body>
</html>"""
    return html_text


def log_importance_weights_viz(
    model_inputs: dict,
    weights: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any = None,
    create_negative_prompt_fn: callable = None,
    weighting_type: str = "importance",
) -> str:
    """Generate HTML visualization for importance weights.
    
    Args:
        model_inputs: Dictionary containing 'responses', 'problem', and other model inputs
        weights: Per-token importance weights (bs, response_length)
        response_mask: Valid token mask (bs, response_length) where 1=valid, 0=padding
        tokenizer: Tokenizer for decoding token IDs (optional)
        create_negative_prompt_fn: Function that takes a prompt and returns a negative prompt (optional)
        weighting_type: Type of importance weighting used (e.g., 'abs', 'kl', '')
    
    Returns:
        HTML string with visualization, or empty string if tokenizer not provided
    """
    if tokenizer is None:
        return ""
    
    try:
        # Find first sample in batch with valid response
        for batch_idx in range(weights.shape[0]):
            num_valid = int(response_mask[batch_idx].sum().item())
            if num_valid == 0:
                continue

            # Extract data for this sample
            response_ids = model_inputs["responses"][batch_idx][:num_valid]
            weight_values = weights[batch_idx][:num_valid].cpu().numpy().tolist() if isinstance(weights, torch.Tensor) else weights[batch_idx][:num_valid]
            
            # Decode tokens
            tokens = tokenizer.convert_ids_to_tokens(response_ids.cpu().tolist() if isinstance(response_ids, torch.Tensor) else response_ids)
            answer_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Get prompts
            problem = model_inputs["problem"][batch_idx] if isinstance(model_inputs["problem"], list) else model_inputs["problem"]
            if not isinstance(problem, str):
                problem = str(problem)
            
            negative_prompt = ""
            if create_negative_prompt_fn is not None:
                negative_prompt = create_negative_prompt_fn(problem)
            
            # Render HTML
            html_content = render_html_heatmap(
                tokens=tokens,
                weights=weight_values,
                question=problem,
                answer=answer_text,
                negative_prompt=negative_prompt,
                metric_name=weighting_type,
            )
            
            # Return HTML content (trainer will handle logging)
            return html_content
    except Exception as e:
        # Return empty string if visualization fails
        return ""
    
    return ""

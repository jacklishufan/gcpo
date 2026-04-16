# test_vllm_qwen3_vl_local_image.py
import base64
import mimetypes
from openai import OpenAI


def local_image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_b64}"


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

image_path = "assets/easyr1_grpo.png"   # change to your local image path
image_data_url = local_image_to_data_url(image_path)

response = client.chat.completions.create(
    model="/home/schmidt/ssci-shufan/scratch_ssci-adityag/Qwen3-VL-8B-Instruct",   # replace if your served model name differs
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ],
        }
    ],
    temperature=0.2,
    max_tokens=256,
)

print(response.choices[0].message.content)
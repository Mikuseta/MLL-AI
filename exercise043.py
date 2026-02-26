import os
import base64
from mimetypes import guess_type
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not deployment:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing in .env")

def local_image_to_data_url(path: str) -> str:
    mime_type, _ = guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64}"

image_path = r"C:\Users\MiklasKulmala\Documents\Praedor\Ghaz.jpg"
image_data_url = local_image_to_data_url(image_path)

resp = client.responses.create(
    model=deployment,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Kuvaile mitä näet tässä kuvassa."},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }
    ],
    max_output_tokens=300,
)

print(resp.output_text)
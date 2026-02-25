import os, base64
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


print("Deployment:", deployment)
def image_to_data_url(path: str) -> str:
    # vaihda mime tarvittaessa: image/jpeg, image/webp...
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

img_url = image_to_data_url(r"C:\Users\MiklasKulmala\Documents\Praedor\Mina.jpeg")

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Kerro mitä kuvassa näkyy ja listaa 5 havaintoa."},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        },
    ],
    temperature=0.2,
    max_tokens=300,
)

print(response.choices[0].message.content)
print("usage:", response.usage.model_dump() if response.usage else None)


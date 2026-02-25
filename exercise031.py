import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "system",
            "content": "You are a precise and analytical history professor."
        },
        {
            "role": "user",
            "content": "Why exactly is the sinking of Titanic such a big deal?"
        }
    ],
    temperature=0.9,
    max_tokens=300
)

print("Assistant reply:\n")
print(response.choices[0].message.content)

print("\nFinish reason:", response.choices[0].finish_reason)
print("Usage:", response.usage.model_dump())
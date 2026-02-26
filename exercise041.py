import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
print("DEPLOYMENT =", deployment)

response = client.responses.create(
    model=deployment,
    input="Explain ork reproduction in Warhammer 40.000.",
)

print(response.output_text)
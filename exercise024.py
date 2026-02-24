from dotenv import load_dotenv
import os
from openai import AzureOpenAI

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([deployment, api_key, endpoint, api_version]):
    raise RuntimeError("Yksi tai useampi env-muuttuja puuttuu.")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
)

response = client.chat.completions.create(
    model=deployment,
    messages=[
  {"role":"system","content":"Vastaa suomeksi. Ole tosi ytimekäs. Käytä bullet pointseja."},
  {"role":"user","content":"Selitä mikä on API."}
    ],
    temperature=0.7,
    max_tokens=200,
)

print(response.choices[0].message.content)
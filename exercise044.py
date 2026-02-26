import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

resp = client.responses.create(
    model=deployment,
    input="Explain what Ork WAAAGH means in Warhammer 40.000."
)

print("Original response:\n", resp.output_text)
print("\nResponse ID:", resp.id)


retrieved = client.responses.retrieve(resp.id)
print("\nRetrieved response:\n", retrieved.output_text)


client.responses.delete(resp.id)
print("\nResponse deleted.")


try:
    client.responses.retrieve(resp.id)
except Exception as e:
    print("\nRetrieving after delete failed (as expected):")
    print(e)
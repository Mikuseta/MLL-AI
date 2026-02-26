import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),  # .../openai/v1/
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not deployment:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing")

# -----------------------------------
# First question
# -----------------------------------
resp1 = client.responses.create(
    model=deployment,
    input="Explain ork society in Warhammer 40.000."
)

print("\nAssistant (1):")
print(resp1.output_text)

# -----------------------------------
# Follow-up referencing previous response
# -----------------------------------
resp2 = client.responses.create(
    model=deployment,
    input="Can you explain that in simpler terms?",
    previous_response_id=resp1.id
)

print("\nAssistant (2):")
print(resp2.output_text)
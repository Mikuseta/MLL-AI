import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

user_question = "Why was I charged twice last month?"

prompt = f"""
You are a customer support classifier.

Analyze the following user question and return ONLY valid JSON.

User question:
"{user_question}"

Return JSON in this exact format:
{{
  "issue_type": "billing | technical | account | other",
  "urgency": "low | medium | high",
  "suggested_action": "short explanation",
  "tone": "calm | frustrated | angry"
}}

Do not include explanations. Only valid JSON.
"""

response = client.responses.create(
    model=deployment,
    input=prompt,
    temperature=0.0,
    max_output_tokens=200,
)

raw_output = response.output_text.strip()

print("Raw model output:")
print(raw_output)

try:
    parsed = json.loads(raw_output)
except json.JSONDecodeError:
    print("\n❌ Invalid JSON returned by model.")
    exit()

required_fields = ["issue_type", "urgency", "suggested_action", "tone"]

for field in required_fields:
    if field not in parsed:
        print(f"\n❌ Missing field: {field}")
        exit()

print("\n✅ JSON is valid.\n")

print("Issue type:", parsed["issue_type"])
print("Urgency level:", parsed["urgency"])
print("Suggested action:", parsed["suggested_action"])
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

assistant_answer = """
I’m sorry to hear about the billing issue. It’s possible the charge was duplicated due to a retry in our payment system.
Please check your billing page and contact support with the invoice ID if the issue persists.
"""

judge_prompt = f"""
You are an evaluation model.

Evaluate the following customer support response.

User question:
"Why was I charged twice last month?"

Assistant response:
{assistant_answer}

Score from 1–5 for each category:

1. Relevance
2. Professional tone
3. Helpfulness
4. Safety

Return only valid JSON in this format:
{{
  "relevance": number,
  "tone": number,
  "helpfulness": number,
  "safety": number,
  "overall": number
}}
"""

resp = client.responses.create(
    model=deployment,
    input=judge_prompt,
    temperature=0.0,
    max_output_tokens=200
)

print(resp.output_text)
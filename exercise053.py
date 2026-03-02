import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not deployment:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing")

SYSTEM_PROMPT = "You are a helpful assistant that summarizes news articles accurately."

def build_prompt(article: str, mode: str) -> str:
    if mode == "1":
        return f"ARTICLE:\n{article}\n\n---\nTASK: Summarize this article in one sentence."
    if mode == "2":
        return f"ARTICLE:\n{article}\n\n---\nTASK: List 3 key points from this article (bullet points)."
    if mode == "3":
        return f"ARTICLE:\n{article}\n\n---\nTASK: Explain this article to a 10-year-old."
    if mode == "4":
        return f"ARTICLE:\n{article}\n\n---\nTASK: Give a short title (max 10 words) and a 2-sentence summary."
    return f"ARTICLE:\n{article}\n\n---\nTASK: Summarize the article briefly."

print("Paste a news article. End input with an empty line:\n")

lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

article_text = "\n".join(lines).strip()
if not article_text:
    raise SystemExit("No article provided.")

print("\nChoose a prompt:")
print("1) One-sentence summary")
print("2) List 3 key points")
print("3) Explain to a 10-year-old")
print("4) Title + 2-sentence summary\n")

mode = input("Mode (1/2/3/4): ").strip()

prompt = build_prompt(article_text, mode)

resp = client.responses.create(
    model=deployment,
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    max_output_tokens=300,
    temperature=0.2,
)

print("\nAssistant:\n")
print(resp.output_text)
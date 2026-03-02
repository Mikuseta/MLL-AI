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

examples = """
Classify the sentiment of each sentence as Positive, Neutral, or Negative.

Example 1:
Sentence: I love this product.
Answer: Positive

Example 2:
Sentence: It's okay, not great.
Answer: Neutral

Example 3:
Sentence: This was terrible.
Answer: Negative
""".strip()

while True:
    user_sentence = input("\nGive a sentence to classify (or type 'exit'): ").strip()
    if user_sentence.lower() == "exit":
        break

    prompt = f"""{examples}

Now classify:
Sentence: {user_sentence}
Answer:"""

    print("\n--- Prompt sent to model ---\n")
    print(prompt)
    print("\n--- Model answer ---\n")

    resp = client.responses.create(
        model=deployment,
        input=prompt,
        max_output_tokens=20,
        temperature=0.0
    )

    print(resp.output_text.strip())
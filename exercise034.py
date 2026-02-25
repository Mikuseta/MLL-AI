import os, time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not deployment:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT puuttuu (.env).")

PRICE_PER_1M_INPUT = 0.0
PRICE_PER_1M_OUTPUT = 0.0

TOKEN_WARN_LIMIT = 3000

messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer clearly and briefly."}
]

print("Chat started. Type 'exit' to quit.\n")

total_prompt_tokens = 0
total_completion_tokens = 0
total_cost_usd = 0.0

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("\nExiting chat.")
        break

    messages.append({"role": "user", "content": user_input})

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
    )
    latency_s = time.perf_counter() - t0

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    if finish_reason != "stop":
        if finish_reason == "length":
            print("\nAssistant: (Response cut off: max_tokens reached.)\n")
        else:
            print(f"\nAssistant: (No response printed. finish_reason={finish_reason})\n")
        continue

    assistant_reply = choice.message.content
    print(f"\nAssistant: {assistant_reply}\n")

    messages.append({"role": "assistant", "content": assistant_reply})

    usage = response.usage
    if usage:
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        cost = (prompt_tokens / 1_000_000) * 2.0 + (completion_tokens / 1_000_000) * 8.0
        total_cost_usd += cost

        print(f"Latency: {latency_s:.2f}s")
        print(f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

        if total_tokens >= TOKEN_WARN_LIMIT:
            print(f"⚠️ Token warning: this call used {total_tokens} tokens (limit={TOKEN_WARN_LIMIT}). Consider shortening context / max_tokens.\n")

        print(f"Estimated cost for this call: ${cost:.6f}")
        print(f"Running totals: prompt={total_prompt_tokens}, completion={total_completion_tokens}, est_cost=${total_cost_usd:.6f}\n")
    else:
        print(f"Latency: {latency_s:.2f}s")
        print("(No usage field returned by API.)\n")
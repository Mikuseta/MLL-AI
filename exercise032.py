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
if not deployment:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT puuttuu (.env).")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer clearly and briefly."
    }
]

print("Multi-turn chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Exiting chat.")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    if finish_reason == "stop":
        assistant_reply = choice.message.content
        print(f"\nAssistant: {assistant_reply}\n")
        messages.append({"role": "assistant", "content": assistant_reply})

    elif finish_reason == "length":
        print("\nAssistant: (Response was cut off because it hit the max token limit.)\n")
        assistant_reply = choice.message.content or ""
        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})

    else:
        print(f"\nAssistant: (No response printed. finish_reason={finish_reason})\n")
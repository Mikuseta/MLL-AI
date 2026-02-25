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
    {"role": "system", "content": "You are a helpful assistant. Answer clearly and briefly."}
]

print("Streaming chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Exiting chat.")
        break

    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
        stream=True,
    )

    print("\nAssistant: ", end="", flush=True)

    assistant_text = ""
    finish_reason = None

    for event in stream:
        if not hasattr(event, "choices") or not event.choices:
            continue

        choice = event.choices[0]

        # Print content chunks as they arrive
        if hasattr(choice, "delta") and choice.delta:
            content = getattr(choice.delta, "content", None)
            if content:
                assistant_text += content
                print(content, end="", flush=True)

        # Capture finish_reason (usually comes at the end)
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    print("\n")  # newline after streaming output

    if finish_reason == "stop":
        messages.append({"role": "assistant", "content": assistant_text})
    elif finish_reason == "length":
        print("(Note: response was cut off due to max_tokens.)\n")
        messages.append({"role": "assistant", "content": assistant_text})
    else:
        print(f"(Note: unexpected finish_reason={finish_reason})\n")
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
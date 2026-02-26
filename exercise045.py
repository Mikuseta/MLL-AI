import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
SYSTEM_PROMPT = "You are a Warhammer 40.000 Ork mekboy. Answer in character."

previous_response_id = None

print("Stateful streaming chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    kwargs = dict(
        model=deployment,
        max_output_tokens=300,
        stream=True,
    )

    if previous_response_id is None:
        kwargs["input"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    else:
        kwargs["input"] = user_input
        kwargs["previous_response_id"] = previous_response_id

    stream = client.responses.create(**kwargs)

    print("\nAssistant: ", end="", flush=True)
    full_text = ""
    final_response_id = None

    for event in stream:
        # Events contain different types; we want "response.output_text.delta" style chunks
        event_dict = event.model_dump() if hasattr(event, "model_dump") else dict(event)

        # Try common fields for text deltas
        if event_dict.get("type") in ("response.output_text.delta", "response.output_text"):
            delta = event_dict.get("delta") or event_dict.get("text") or ""
            if delta:
                full_text += delta
                print(delta, end="", flush=True)

        # Capture the response id when available
        if "response" in event_dict and isinstance(event_dict["response"], dict):
            final_response_id = event_dict["response"].get("id") or final_response_id

        # Sometimes id is at top-level
        final_response_id = event_dict.get("response_id") or final_response_id

    print("\n")
    previous_response_id = final_response_id or previous_response_id
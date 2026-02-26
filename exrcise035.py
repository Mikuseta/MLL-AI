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


def extract_content_filter_info(response) -> dict:
    data = response.model_dump() if hasattr(response, "model_dump") else response
    info = {
        "prompt_filter_results": data.get("prompt_filter_results"),
        "output_filter_results": None,
    }
    choices = data.get("choices", [])
    if choices:
        info["output_filter_results"] = choices[0].get("content_filter_results")
    return info


def should_block_response(filter_info: dict, block_severities=("high",), block_if_filtered=True):
    output = filter_info.get("output_filter_results")
    if not output:
        return (False, "No content_filter_results present.")

    reasons = []
    for category, result in output.items():
        if not isinstance(result, dict):
            continue
        filtered = result.get("filtered")
        severity = result.get("severity")
        if block_if_filtered and filtered is True:
            reasons.append(f"{category}: filtered=True")
        if severity and str(severity).lower() in [s.lower() for s in block_severities]:
            reasons.append(f"{category}: severity={severity}")

    if reasons:
        return (True, "; ".join(reasons))
    return (False, "OK")


messages = [{"role": "system", "content": "You are an ork warboss in Warhammer 40.000 leading a Waaaagh, apply this to your replies. You are violent and hopeful in your responses"}]
print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
    )
    latency = time.perf_counter() - t0

    filter_info = extract_content_filter_info(response)
    block, reason = should_block_response(
        filter_info,
        block_severities=("high",),   
        block_if_filtered=True
    )

    if block:
        print("\nAssistant: Sorry, I can’t provide an answer to that.\n")
        print(f"(Blocked by content filter: {reason})\n")
        continue

    reply = response.choices[0].message.content
    print(f"\nAssistant: {reply}\n")
    print(f"Latency: {latency:.2f}s")

    messages.append({"role": "assistant", "content": reply})
import tiktoken

def num_tokens_from_string(text: str, encoding_name: str) -> int:
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(enc.encode(text))

def compare_encodings(example_string: str) -> None:
    print(f"\nExample string: {example_string}")

    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        try:
            enc = tiktoken.get_encoding(encoding_name)
        except KeyError:
            print(f"\n{encoding_name}: ei löydy tässä tiktoken-versiossa")
            continue

        token_integers = enc.encode(example_string)
        token_bytes = [enc.decode_single_token_bytes(t) for t in token_integers]

        print(f"\n{encoding_name}: {len(token_integers)} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")

if __name__ == "__main__":
    print(num_tokens_from_string("tiktoken is great!", "cl100k_base"))
    compare_encodings("antidisestablishmentarianism")

def token_count(text: str, enc_name: str) -> int:
    enc = tiktoken.get_encoding(enc_name)
    return len(enc.encode(text))

def compare_token_counts(text: str, encodings=None):
    if encodings is None:
        encodings = ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]

    print(f"Text length (chars): {len(text)}")
    for name in encodings:
        try:
            n = token_count(text, name)
            print(f"{name:12s}: {n} tokens")
        except KeyError:
            print(f"{name:12s}: (not available in your tiktoken)")

sample = "Jos haet poikatyttömäistä, käytännönläheistä ja vähän suorapuheista, näistä voisi löytyä parempi sana " * 50
compare_token_counts(sample)

enc = tiktoken.get_encoding("cl100k_base")

target = 128_000
chunk = "This is one sentence to fill the context. "
text = ""

while True:
    # kokeillaan lisätä 100 lausetta kerralla (nopeampi kuin 1 kerrallaan)
    candidate = text + chunk * 100
    n = len(enc.encode(candidate))

    if n >= target:
        break
    text = candidate

final_tokens = len(enc.encode(text))
print("Tokens:", final_tokens)
print("Characters:", len(text))
print("Approx words:", len(text.split()))
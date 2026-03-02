import os
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBED_MODEL = "text-embedding-ada-002"  # sama kuin sulla aiemmin

# --- 1) "Dataset" eli dokumentit ---
docs = [
    {
        "id": "doc-001",
        "title": "Resetting your NimbusCRM password",
        "text": "If you forgot your password, use the 'Forgot password' link on the login page. "
                "You will receive an email to set a new password. For security reasons, support cannot reset it for you."
    },
    {
        "id": "doc-002",
        "title": "Why you might see a duplicate charge",
        "text": "Duplicate charges can happen if a payment attempt is retried by your bank or payment processor. "
                "Check your invoices in Billing. If you see two invoices for the same period, contact support with invoice IDs."
    },
    {
        "id": "doc-003",
        "title": "Troubleshooting service outages",
        "text": "If NimbusCRM seems down, first check our status page and incident updates. "
                "Try logging out and back in, and check your network. If the issue persists, report the time and error message."
    },
    {
        "id": "doc-004",
        "title": "Managing API keys safely",
        "text": "Keep API keys secret and rotate them regularly. Never share keys in chat or email. "
                "If you suspect a key leak, revoke the key immediately and create a new one in the dashboard."
    },
    {
        "id": "doc-005",
        "title": "Canceling your subscription",
        "text": "To cancel, go to Settings → Billing → Subscription and choose Cancel. "
                "Your access remains active until the end of the billing period. Download your data before the end date."
    },
]

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# --- 2) Esilasketaan dokumenttien embeddingit (tehokkaampaa kuin joka haussa) ---
doc_embeddings = []
for d in docs:
    emb = get_embedding(d["title"] + "\n" + d["text"])
    doc_embeddings.append(emb)

doc_embeddings = np.array(doc_embeddings)

def retrieve_top_k(query: str, k: int = 3):
    q_emb = np.array(get_embedding(query)).reshape(1, -1)
    sims = cosine_similarity(q_emb, doc_embeddings)[0]  # shape: (num_docs,)
    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for i in top_idx:
        results.append({
            "score": float(sims[i]),
            "id": docs[i]["id"],
            "title": docs[i]["title"],
            "text": docs[i]["text"],
        })
    return results

if __name__ == "__main__":
    print("Retrieval demo (no LLM). Type 'exit' to quit.\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() == "exit":
            break

        hits = retrieve_top_k(query, k=3)

        print("\nTop 3 documents:\n")
        for rank, h in enumerate(hits, start=1):
            print(f"{rank}) {h['title']}  (score={h['score']:.3f}, id={h['id']})")
            print(f"   {h['text']}\n")
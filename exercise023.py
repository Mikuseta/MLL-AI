import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

sentences = [
    "No indications of panic, shame, or moral conflict.",
    "Not the one she remembered from before Kendachi, before the blood and the blades.",
    "Shows territorial behavior regarding performance space and audience attention.",
    "You make me want to stop killing things.",
    "Eye contact is steady, analytical, and lacking reactive empathy.",
    "Enroll client in individual performance mentorship to channel ambition without interpersonal harm.",
    "Implement strict supervision in any environment involving tools or competitive artistic activity.",
    "Begin empathy conditioning curriculum, with focus on understanding others as independent actors, not rivals.",
    "Parents strongly encouraged to reduce competitive framing of the client’s musical development.",
    "The words hit harder than any bullet.",
    "The way her eyes softened when she looked at Akiko.",
    "Just the rhythm of breathing and static jazz drifting from the radio."
]

# 🔧 tärkeä: käytä embedding-deploymentin nimeä
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

embeddings = []
for text in sentences:
    resp = client.embeddings.create(
        input=text,
        model=embedding_deployment
    )
    embeddings.append(resp.data[0].embedding)

X = np.array(embeddings, dtype=float)

# Similarity
sim_matrix = cosine_similarity(X)
print(np.round(sim_matrix, 2))

# PCA plot
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
for i, text in enumerate(sentences):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    label = text[:40] + "..." if len(text) > 40 else text
    plt.annotate(label, (reduced[i, 0] + 0.01, reduced[i, 1] + 0.01), fontsize=8)

plt.title("Sentence Embeddings (PCA Projection)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()
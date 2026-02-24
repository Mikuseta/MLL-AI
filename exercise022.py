from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([deployment, api_key, endpoint, api_version]):
    raise RuntimeError("Yksi tai useampi env-muuttuja puuttuu.")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
)

sentences = [
    "No indications of panic, shame, or moral conflict.",
    "Not the one she remembered from before Kendachi, before the blood and the blades.",
    "Shows territorial behavior regarding performance space and audience attention.",
    "You make me want to stop killing things.",
    "Eye contact is steady, analytical, and lacking reactive empathy."
]

embeddings = []
for text in sentences:
    response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
    )
    embeddings.append(response.data[0].embedding)

sim_matrix = cosine_similarity(embeddings)
print(np.round(sim_matrix, 2))
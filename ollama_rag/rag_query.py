import requests
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedder & Chroma
embedder = SentenceTransformer(
    'all-MiniLM-L6-v2',
    trust_remote_code=False  # default, but explicit
)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("rag_docs")

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1"

def retrieve(query: str, top_k: int = 2):
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )
    # Concatenate retrieved chunks
    docs = results["documents"][0]
    return "\n\n".join(docs)

def ask_llm(question: str, context: str) -> str:
    prompt = f"""Use only the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise Exception(f"Ollama error: {response.text}")

# Example usage
if __name__ == "__main__":
    question = input("Ask a question: ")
    context = retrieve(question)
    print("\n[Retrieved Context]:")
    print(context)
    print("\n[Answer]:")
    print(ask_llm(question, context))
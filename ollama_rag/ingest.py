import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize embedding model (runs locally)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma client (saves to ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")

# Load documents
docs = []
ids = []
filenames = []

for i, filepath in enumerate(glob.glob("docs/*.txt")):
    with open(filepath, "r") as f:
        text = f.read()
        docs.append(text)
        ids.append(f"doc_{i}")
        filenames.append(os.path.basename(filepath))

# Generate embeddings
print("Generating embeddings...")
embeddings = embedder.encode(docs).tolist()

# Store in Chroma
collection.add(
    embeddings=embeddings,
    documents=docs,
    metadatas=[{"source": name} for name in filenames],
    ids=ids
)

print(f"✅ Ingested {len(docs)} documents into ChromaDB.")

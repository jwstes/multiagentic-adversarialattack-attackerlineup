from qdrant_client import QdrantClient, models
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

qdrant_client = QdrantClient(
    url="https://5cf05368-4850-4f7a-8912-f2af8b6b788c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rQVnGexdv6IYTOvcOEN5Q7EEo9Vq3wuQCwdibak7eIw",
)

COLLECTION_NAME = "ART-Toolbox-AttackStrategies"
PDF_DIRECTORY = "allPDFs"
SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
BATCH_SIZE = 128

def extract_text_from_pdfs(pdf_directory: str) -> str:
    text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text()
    return text

def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER)
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    return embeddings

# def store_in_qdrant(qdrant_client: QdrantClient, collection_name: str, embeddings: list[list[float]], chunks: list[str]):
#     vector_size = len(embeddings[0])

#     qdrant_client.recreate_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
#     )

#     qdrant_client.upsert(
#         collection_name=collection_name,
#         points=models.Batch(
#             ids=list(range(len(chunks))),
#             vectors=embeddings,
#             payloads=[{"text": chunk} for chunk in chunks]
#         ),
#         wait=True
#     )

def store_in_qdrant_batched(qdrant_client: QdrantClient, collection_name: str, embeddings: list[list[float]], chunks: list[str]):
    vector_size = len(embeddings[0])

    collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist. Creating it.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")

    total_chunks = len(chunks)
    print(f"Upserting {total_chunks} points in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, total_chunks, BATCH_SIZE)):
        start_idx = i
        end_idx = min(i + BATCH_SIZE, total_chunks)

        qdrant_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(start_idx, end_idx)),
                vectors=embeddings[start_idx:end_idx],
                payloads=[{"text": chunk} for chunk in chunks[start_idx:end_idx]]
            ),
            wait=True
        )
    print("Upserting complete.")

print("Extracting Text from PDFs")
text = extract_text_from_pdfs(PDF_DIRECTORY)
print("Chunking Text")
chunks = chunk_text(text)
print("Generating Embeddings")
embeddings = generate_embeddings(chunks)

print("Storing in Qdrant")
# store_in_qdrant(qdrant_client, COLLECTION_NAME, embeddings, chunks)
store_in_qdrant_batched(qdrant_client, COLLECTION_NAME, embeddings, chunks)
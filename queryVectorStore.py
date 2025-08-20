from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

qdrant_client = QdrantClient(
    url="https://5cf05368-4850-4f7a-8912-f2af8b6b788c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rQVnGexdv6IYTOvcOEN5Q7EEo9Vq3wuQCwdibak7eIw",
)

COLLECTION_NAME = "ART-Toolbox-AttackStrategies"
PDF_DIRECTORY = "allPDFs"
SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
BATCH_SIZE = 128

def retrieve_context(qdrant_client: QdrantClient, collection_name: str, query: str, top_k: int = 5) -> str:
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER)
    query_embedding = embedding_model.encode(query).tolist()

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    context = ""
    for result in search_result:
        context += result.payload['text'] + "\n---\n"
    return context

def query_vector_store(query: str):
    context = retrieve_context(qdrant_client, COLLECTION_NAME, query)
    print(context)

# user_query = "black-box adversarial attacks for high-resolution images query-efficient score-based"
# query_vector_store(user_query)
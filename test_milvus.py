from pymilvus import MilvusClient
client = MilvusClient("rag_project.db")
if client.has_collection(collection_name="rag_project_collection"):
    client.drop_collection(collection_name="rag_project_collection")
client.create_collection(
    collection_name="rag_project_collection",
    dimension=768,
)

from langchain_milvus import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "./rag_project.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    auto_id=True 
)
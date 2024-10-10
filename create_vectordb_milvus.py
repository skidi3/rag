import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from milvus_db import vector_store as milvus_vector_store

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print(os.getenv("OPENAI_API_KEY"))

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
print(embeddings)

# Load and process the code files
code_files = []
for file in os.listdir("code_files"):
    if file.endswith(".py") | file.endswith(".txt"):  # We're only processing Python files here
        loader = TextLoader(f"code_files/{file}")
        code_files.extend(loader.load())

# Split the code into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(code_files)

print(docs)

# Use the Milvus vector store (defined in the milvus_db file)
db = milvus_vector_store

# Add documents to the Milvus vector store
db.add_documents(docs)

print("Vector database created and stored in Milvus.")

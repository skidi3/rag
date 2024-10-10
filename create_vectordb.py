import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# from milvus_db import vector_store as milvus_vector_store



# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print(os.getenv("OPENAI_API_KEY"))


# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
print(embeddings)
print(embeddings)


# Load and process the code files
code_files = []
for file in os.listdir("code_files"):
    if file.endswith(".py"):  # We're only processing Python files here
        loader = TextLoader(f"code_files/{file}")
        code_files.extend(loader.load())


#print("code files", code_files)
# Split the code into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(code_files)

print(docs)



# Create and save the vector database
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("Vector database created and saved.")
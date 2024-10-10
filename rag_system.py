import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
# from milvus_db import vector_store as milvus_vector_store

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load the vector database we created earlier
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize the OpenAI language model
llm = OpenAI(temperature=0)

# Create a retrieval-based question-answering system
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

def ask_question(question):
    """Function to ask a question and get an answer"""
    return qa.run(question)

# Main loop for interacting with the system
if __name__ == "__main__":
    print("Welcome to the Code Q&A System!")
    print("You can ask questions about the code in your 'code_files' folder.")
    print("Type 'quit' to exit the program.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'quit':
            print("Thank you for using the Code Q&A System. Goodbye!")
            break
        answer = ask_question(user_question)
        print(f"\nAnswer: {answer}")
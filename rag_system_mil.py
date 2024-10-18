import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from milvus_db import vector_store as milvus_vector_store

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Use the Milvus vector store (defined in the milvus_db file)
db = milvus_vector_store

# Initialize two OpenAI language models: one for RAG and one for fallback
llm_rag = OpenAI(temperature=1)  # Lower temperature for more precise answers when using RAG
llm_fallback = OpenAI(temperature=0.8)  # Higher temperature for more flexible fallback answers

# Create a prompt template for RAG
rag_prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't have enough information to answer the question, say "I don't have enough information to answer that question."

Context: {context}
Question: {question}

Answer: """

RAG_PROMPT = PromptTemplate(
    template=rag_prompt_template, input_variables=["context", "question"]
)

# Create a retrieval-based question-answering system
qa = RetrievalQA.from_chain_type(
    llm=llm_rag,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

def ask_question(question):
    """Function to ask a question and get an answer"""
    # First, try to answer using the RAG system
    rag_result = qa({"query": question})
    
    if rag_result['source_documents'] and "I don't have enough information" not in rag_result['result']:
        return f"Answer based on our smartphone & coding database: {rag_result['result']}"
    else:
        # If RAG doesn't have an answer, use the fallback LLM
        fallback_prompt = f"You are a helpful AI assistant knowledgeable about smartphones and coding. Please answer the following question to the best of your ability: {question} with proper text or code and syntax if neccessary."
        fallback_result = llm_fallback(fallback_prompt)
        return f"I don't have specific information about that in my database. Here's a general response: {fallback_result}"

# Main loop for interacting with the system
if __name__ == "__main__":
    print("Welcome to the Smartphone & Coding Q&A System!")
    print("You can ask questions about various smartphone models and programming.")
    print("I'll use my smartphone & coding database when possible, or provide a general answer if needed.")
    print("Type 'quit' to exit the program.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'quit':
            print("Thank you for using the Smartphone & coding Q&A System. Goodbye!")
            break
        answer = ask_question(user_question)
        print(f"\n{answer}")
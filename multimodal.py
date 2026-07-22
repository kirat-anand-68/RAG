from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os
from dotenv import load_dotenv

# from chromaDB import vectorstore

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

loader = TextLoader("langchain_rag_dataset.txt")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)
print(chunks)

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
## step3: Create MMR Retriever
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
# Step 4: Prompt and LLM
prompt = PromptTemplate.from_template("""
Answer the question based on the context provided.

Context:
{context}

Question: {input}
""")
llm = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2
)
# Step 5: RAG Pipeline
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# Step 6: Query
query = {"input": "How does MMR works?"}
response = rag_chain.invoke(query)

print("✅ Answer:\n", response["answer"])

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

## load the text file
loader=TextLoader("langchain_sample.txt")
raw_docs=loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

## user query
query="How can i use the langchain to build an application with memory and tools?"
### FAISS and huggingface mmodel embeddings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(docs,embedding_model)
retriever=vectorstore.as_retriever(search_kwargs={"k":8})
print(retriever)

## prompt and use the llm
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)
print(llm)
# Prompt Template
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
""")
retrieved_docs=retriever.invoke(query)
print(retrieved_docs)

chain=prompt | llm |StrOutputParser()
print(chain)

doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
formatted_docs = "\n".join(doc_lines)

response = chain.invoke({"question":query,"documents":formatted_docs})
print(response)

indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
print(indices)
reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
print(reranked_docs)

# Step 6: Show results
print("\n📊 Final Reranked Results:")
for i, doc in enumerate(reranked_docs, 1):
    print(f"\nRank {i}:\n{doc.page_content}")

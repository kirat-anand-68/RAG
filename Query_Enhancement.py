from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

## step1 : Load and split the dataset
loader = TextLoader("langchain_crewai_dataset.txt")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

print(chunks)
### step 2: Vector Store
embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(chunks,embedding_model)

## step 3:MMR Retriever
retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":5})
print(retriever)

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2
)
print(llm)
# Query expansion
query_expansion_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.

Original query: "{query}"

Expanded query:
""")

query_expansion_chain=query_expansion_prompt| llm | StrOutputParser()
print(query_expansion_chain)

print(query_expansion_chain.invoke({"query":"Langchain memory"}))

# RAG answering prompt
answer_prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")

document_chain=create_stuff_documents_chain(llm=llm,prompt=answer_prompt)
rag_pipeline = (
    RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: retriever.invoke(query_expansion_chain.invoke({"query": x["input"]}))
    })
    | document_chain
)
# Step 6: Run query
query = {"input": "What types of memory does LangChain support?"}
print(query_expansion_chain.invoke({"query":query}))
response = rag_pipeline.invoke(query)
print("✅ Answer:\n", response)

# Step 6: Run query
query = {"input": "CrewAI agents?"}
print(query_expansion_chain.invoke({"query":query}))
response = rag_pipeline.invoke(query)
print("✅ Answer:\n", response)

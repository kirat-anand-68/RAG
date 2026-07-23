from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load and chunk your dataset
chunk_size = 300
chunk_overlap = 100

# loading data
loader = WikipediaLoader(query="Steve Jobs", load_max_docs=5)
documents = loader.load()

# text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
docs = text_splitter.split_documents(documents=documents)
print(docs)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2
)
from langchain_community.vectorstores import Chroma
## creating vector store
db = Chroma.from_documents(documents = docs,embedding=embeddings,persist_directory = "output/steve_jobs_for_hyde.db")
##create the retriever
base_retriever=db.as_retriever(search_kwargs = {"k":5})
from langchain_core.output_parsers import StrOutputParser
## Generating a prompt for generating a hyde
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
def get_hyde_doc(query):
    template = """Imagine you are an expert writing a detailed explanation on the topic: '{query}'
    create a hypothetical answer for the topic"""


    system_message_prompt = SystemMessagePromptTemplate.from_template(template = template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    messages = chat_prompt.format_prompt(query = query).to_messages()
    print(messages)
    response = llm.invoke(messages)
    hypo_doc = response.content
    return hypo_doc
query = 'When was Steve Jobs fired from Apple?'
print(get_hyde_doc(query=query))
matched_doc = base_retriever.invoke(get_hyde_doc(query))
print(matched_doc)

## Langchain -Hypothetical document embedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Step 1: Load documents
loader = TextLoader("langchain_crewai_dataset.txt")
docs = loader.load()

# Step 2: Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
print(chunks)
base_embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Step 3: HyDE Embedder using prompt_key='web_search'
hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"
)
# Step 4: Store documents in Chroma with HyDE embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hyde_embedding_function,
    persist_directory="output/langchain"
)
# Step 5: RAG answer generation prompt
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
rag_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
# Step 6: Final RAG Pipeline
def hyde_rag_pipeline(query):
    matched_docs = vectorstore.similarity_search(query, k=4)
    print(matched_docs)
    response = rag_chain.invoke({
        "input": query,
        "context": matched_docs
    })
    return response
# Step 7: Run example query
query = "What memory modules does LangChain provide?"
answer = hyde_rag_pipeline(query)
print("✅ Final Answer:\n", answer)

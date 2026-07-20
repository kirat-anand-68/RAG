import os
from dotenv import load_dotenv
import numpy as np
import warnings

# from chromaDB import vectorstore

warnings.filterwarnings('ignore')

# LangChain core imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,

)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# LangChain specific imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

## Data Ingestion and processing
## i use the document datastructiure
sample_documents = [
    Document(
        page_content="""
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        These systems are designed to think like humans and mimic their actions.
        AI can be categorized into narrow AI and general AI.
        """,
        metadata={"source": "AI Introduction", "page": 1, "topic": "AI"}
    ),
    Document(
        page_content="""
        Machine Learning is a subset of AI that enables systems to learn from data.
        Instead of being explicitly programmed, ML algorithms find patterns in data.
        Common types include supervised, unsupervised, and reinforcement learning.
        """,
        metadata={"source": "ML Basics", "page": 1, "topic": "ML"}
    ),
    Document(
        page_content="""
        Deep Learning is a subset of machine learning based on artificial neural networks.
        It uses multiple layers to progressively extract higher-level features from raw input.
        Deep learning has revolutionized computer vision, NLP, and speech recognition.
        """,
        metadata={"source": "Deep Learning", "page": 1, "topic": "DL"}
    ),
    Document(
        page_content="""
        Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
        It combines computational linguistics with machine learning and deep learning models.
        Applications include chatbots, translation, sentiment analysis, and text summarization.
        """,
        metadata={"source": "NLP Overview", "page": 1, "topic": "NLP"}
    )
]

print(sample_documents)

## test splittning
text_Splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[" "]
)

## Split the documents into chunks
chunks = text_Splitter.split_documents(sample_documents)
print(chunks)

## load the embeddings models
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# create an embedding for a single text
sample_text = "What is machine learning"

sample_embedding = embeddings.embed_query(sample_text)
print(sample_embedding)
print("Embedding dimensions:", len(sample_embedding))

texts=["AI","Machine learning","Deep Learning","Neural Networks"]
batch_embeddings=embeddings.embed_documents(texts)
print(batch_embeddings[0])

## compare the embeddings usng the cosine similarity
def compare_embeddings(text1:str,text2:str):
    """Compare semantic similarity of 2 texts using embeddings"""

    emb1=np.array(embeddings.embed_query(text1))
    emb2=np.array(embeddings.embed_query(text2))

    ## calculate the similarity score

    similarity=np.dot(emb1, emb2) / (np.linalg.norm(emb1)* np.linalg.norm(emb2))
    return similarity
print("\nSemantic Similarity Examples:")
print(f"'AI' vs 'Artificial Intelligence': {compare_embeddings('AI', 'Artificial Intelligence'):.3f}")

## Create FAISS vectorstore

vectorstore=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print(f"Vector store created with {vectorstore.index.ntotal} vectors")

## Save Vector stor for later use
vectorstore.save_local("faiss_index")
print("Vector store saved to 'faiss_index' directory")

## load the vector store
loaded_vectorstore=FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Loaded vector store contains {loaded_vectorstore.index.ntotal} vectors")

## Similarity Search
query="What is deep learninng"

results=vectorstore.similarity_search(query,k=3)
print(results)

print(f"Query: {query}\n")
print("Top 3 similar chunks:")
for i, doc in enumerate(results):
    print(f"\n{i+1}. Source: {doc.metadata['source']}")
    print(f"   Content: {doc.page_content[:200]}...")

results_with_scores=vectorstore.similarity_search_with_score(query,k=3)

print("\n\nSimilarity search with scores:")
for doc, score in results_with_scores:
    print(f"\nScore: {score:.3f}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:100]}...")

filter_dict={"topic":"ML"}
filtered_results=vectorstore.similarity_search(
    query,
    k=3,
    filter=filter_dict
)
print(filtered_results)
print(len(filtered_results))


##RAG CHAIN WITH LCEL

## LLM groq LLM
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=init_chat_model(model="groq:llama-3.3-70b-versatile")
print(llm)
simple_prompt=ChatPromptTemplate.from_template("""answer the question based only on the following context:
Context: {context}

Question: {question}
Answer:""")

retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
print(retriever)

from typing import List
# Format documents for the prompt
def format_docs(docs: List[Document]) -> str:
    """Format documents for insertion into prompt"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)

simple_rag_chain=(
    {"context":retriever | format_docs,"question":RunnablePassthrough() }
    | simple_prompt
    | llm
    |StrOutputParser()

)

print(simple_rag_chain)


## conversational rag chain
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the provided context to answer questions."),
    ("placeholder", "{chat_history}"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])
def create_conversational_rag():
    """Create a conversational RAG chain with memory"""
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | conversational_prompt
        | llm
        | StrOutputParser()
    )

conversational_rag = create_conversational_rag()
print(conversational_rag)

### Streaming RAG chain
streaming_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
)
print("Modern RAG chains created successfully!")
print("Available chains:")
print("- simple_rag_chain: Basic Q&A")
print("- conversational_rag: Maintains conversation history")
print("- streaming_rag_chain: Supports token streaming")


# Test function for different chain types
def test_rag_chains(question: str):
    """Test all RAG chain variants"""
    print(f"Question: {question}")
    print("=" * 80)

    # 1. Simple RAG
    print("\n1. Simple RAG Chain:")
    answer = simple_rag_chain.invoke(question)
    print(f"Answer: {answer}")

    print("\n2. Streaming RAG:")
    print("Answer: ", end="", flush=True)
    for chunk in streaming_rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print()
test_rag_chains("What is the difference between AI and machine learning")
print(test_rag_chains)

## Conversational example
print("\n3. Conversational RAG Example:")
chat_history = []

# First question
q1 = "What is machine learning?"
a1 = conversational_rag.invoke({
    "input": q1,
    "chat_history": chat_history
})

print(f"Q1: {q1}")
print(f"A1: {a1}")

chat_history.extend([
    HumanMessage(content=q1),
    AIMessage(content=a1)
])

# Follow-up question
q2 = "How is it different from traditional programming?"
a2 = conversational_rag.invoke({
    "input": q2,
    "chat_history": chat_history
})
print(f"\nQ2: {q2}")
print(f"A2: {a2}")

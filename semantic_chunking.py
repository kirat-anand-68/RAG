from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.testing.suite.test_reflection import metadata

# from chromaDB import vectorstore

## Document segmentation convert into text and paragraphs
model=SentenceTransformer('all-MiniLM-L6-v2')

# sample_test
text="""
LangChain is a framework for building applications with LLMs.
Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone.
You can create chains, agents, memory, and retrievers.
The Eiffel Tower is located in Paris.
France is a popular tourist destination.
"""

## Step 1: split into sentences
sentences=[s.strip() for s in text.split("\n")if s.strip()]
## step 2: Embed each setence
embeddings=model.encode(sentences)
# Step 3: Initialize parameters
threshold = 0.7  # control chunk tightness
chunks = []
current_chunk=[sentences[0]]

## Step 4: Semantic grouping based on threshold
for i in range(1, len(sentences)):
    sim = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    if sim>=threshold:
        current_chunk.append(sentences[i])
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk=[sentences[i]]


# Append the last chunk
chunks.append(" ".join(current_chunk))

# Output the chunks
print("\n📌 Semantic Chunks:")
for idx, chunk in enumerate(chunks):
    print(f"\nChunk {idx+1}:\n{chunk}")

## RAG pipeline with semantic chunks
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_classic.schema import Document
from langchain_classic.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.schema.runnable import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv("../.env")
# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

### custom semantic chunker


class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model=SentenceTransformer(model_name)
        self.threshold=threshold

    def split(self, text: str):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def split_documents(self,docs):
        result=[]
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))

        return result
# Sample text
sample_text = """
LangChain is a framework for building applications with LLMs.
Langchain provides modular abstractions to combine LLMs with tools like OpenAI and Pinecone.
You can create chains, agents, memory, and retrievers.
The Eiffel Tower is located in Paris.
France is a popular tourist destination.
"""

doc = Document(page_content=sample_text)
print(doc)

### chunking
chunker=ThresholdSemanticChunker(threshold=0.7)
chunks=chunker.split_documents([doc])
print(chunks)

## vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embedding
)

retriever = vectorstore.as_retriever()

print(retriever)
### prompt template
## Prompt Template

# --- 5. Prompt Template ---
template = """Answer the question based on the following context:

{context}

Question: {question}
"""

prompt=PromptTemplate.from_template(template)
print(prompt)

llm = init_chat_model(
    model="groq:llama-3.3-70b-versatile",
    temperature=0.4
)
### LCEL Chain With retrieval

rag_chain=(
    RunnableMap(
        {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- 8. Run Query ---
# query = {"question": "What is LangChain used for?"}
# result = rag_chain.invoke(query)
#
# print(result)
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.document_loaders import TextLoader
#
# loader=TextLoader("langchain_intro.txt")
# docs=loader.load()
#
# ## Initialize embedding model
# embedding=OpenAIEmbeddings()
#
# ## Create the semantic chunker
# chunker=SemanticChunker(embedding)
#
# ## Split the documents
# chunks=chunker.split_documents(docs)
#
# ## Result
#
# for i,chunk in enumerate(chunks):
#     print(f"\n chunk {i+1}:\n{chunk.page_content}")


import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel,Field

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------------
# 1. Prepare Vector Store
# -------------------------------

docs = TextLoader(
    "research_notes.txt",
    encoding="utf-8"
).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever = vectorstore.as_retriever()

# -------------------------------
# 2. Initialize LLM
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------------
# 3. LangGraph State Definition
# -------------------------------

class RAGCoTState(BaseModel):
    question: str
    sub_steps: List[str] = Field(default_factory=list)
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""

def plan_steps(state:RAGCoTState)->RAGCoTState:
    prompt=f"Break the question into 2-3 reasoning steps: \n\n {state.question}"
    result=llm.invoke(prompt).content
    subqs=[line.strip("-")for line in result.split("\n") if line.strip()]

    return state.model_copy(update={"sub_steps":subqs})
def retrieve_per_step(state:RAGCoTState)->RAGCoTState:
    all_docs=[]
    for sub in state.sub_steps:
        docs=retriever.invoke(sub)
        all_docs.extend(docs)
    return state.model_copy(update={"retrieved_docs":all_docs})
# Generate the answer
# c. Generate Final Answer
def generate_answer(state: RAGCoTState) -> RAGCoTState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""
You are answering a complex question using reasoning and retrieved documents.

Question: {state.question}

Relevant Information:
{context}

Now synthesize a well-reasoned final answer.
"""
    result = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": result})

# -------------------------------
# 4. LangGraph Graph
# -------------------------------
builder = StateGraph(RAGCoTState)
builder.add_node("planner", plan_steps)
builder.add_node("retriever", retrieve_per_step)
builder.add_node("responder", generate_answer)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO

graph_png = graph.get_graph().draw_mermaid_png()

image = mpimg.imread(BytesIO(graph_png), format="png")

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
# -------------------------------
# 5. Run CoT RAG Agent
# -------------------------------
if __name__ == "__main__":
    query = "what are the additional eperiments in Transformer evaluation?"
    state = RAGCoTState(question=query)
    final = graph.invoke(state)

    print("\n🪜 Reasoning Steps:", final["sub_steps"])
    print("\n✅ Final Answer:\n", final["answer"])

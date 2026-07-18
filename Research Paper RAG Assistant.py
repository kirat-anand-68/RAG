import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from pymupdf.extra import page_count
# from sqlalchemy.testing.suite.test_reflection import metadata

#___________________________________________________________
# Load the environment
#________________________________________________________________
load_dotenv()
# from chromaDB import vectorstore
# streamlit page configuration
st.set_page_config(
    page_title="Research paper RAG Assistant",
    page_icon="📚",
    layout="wide"
)
# =========================================================================
#CUSTOM CSS
# ============================================================================
st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 20%,
            rgba(50, 90, 255, 0.16),
            transparent 30%),
        radial-gradient(circle at 90% 30%,
            rgba(140, 60, 255, 0.14),
            transparent 35%),
        linear-gradient(
            135deg,
            #050816 0%,
            #080c1c 50%,
            #050712 100%
        );

    background-attachment: fixed;
    color: white;
}


/* ---------- MAIN CONTENT ---------- */

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ---------- MAIN TITLE ---------- */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -1px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #7da6ff,
        #a78bfa,
        #67e8f9,
        #ffffff
    );

    background-size: 300%;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: titleAnimation 7s ease infinite;
}


@keyframes titleAnimation {

    0% {
        background-position: 0%;
    }

    50% {
        background-position: 100%;
    }

    100% {
        background-position: 0%;
    }
}


/* ---------- SUBTITLE ---------- */

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 17px;
    margin-top: 8px;
    margin-bottom: 40px;
}


/* ---------- RESEARCH BADGE ---------- */

.research-badge {

    display: inline-block;

    padding: 7px 16px;

    border-radius: 30px;

    background:
        rgba(100, 120, 255, 0.10);

    border:
        1px solid rgba(120, 150, 255, 0.25);

    color: #9db8ff;

    font-size: 12px;

    letter-spacing: 1px;

    box-shadow:
        0 0 25px rgba(80, 100, 255, 0.10);
}


/* ---------- CHAT MESSAGES ---------- */

[data-testid="stChatMessage"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.07),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.10);

    backdrop-filter: blur(20px);

    border-radius: 18px;

    padding: 18px;

    margin-bottom: 15px;

    box-shadow:
        0 10px 40px rgba(0,0,0,0.30);

    transition:
        transform 0.3s ease,
        box-shadow 0.3s ease,
        border 0.3s ease;
}


[data-testid="stChatMessage"]:hover {

    transform: translateY(-3px);

    border:
        1px solid rgba(100,140,255,0.35);

    box-shadow:
        0 15px 50px rgba(0,0,0,0.40),
        0 0 30px rgba(80,120,255,0.10);
}


/* ---------- CHAT INPUT ---------- */

[data-testid="stChatInput"] {

    background:
        rgba(8,12,30,0.85);

    backdrop-filter: blur(25px);

    border-radius: 18px;

    border:
        1px solid rgba(120,150,255,0.20);

    box-shadow:
        0 10px 40px rgba(0,0,0,0.5),
        0 0 30px rgba(80,120,255,0.10);
}


/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(8,12,28,0.98),
            rgba(5,8,20,0.98)
        );

    border-right:
        1px solid rgba(255,255,255,0.08);

    box-shadow:
        10px 0 40px rgba(0,0,0,0.30);
}


/* ---------- SIDEBAR HEADINGS ---------- */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #8ab4ff
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


/* ---------- BUTTONS ---------- */

.stButton > button {

    width: 100%;

    background:
        linear-gradient(
            135deg,
            #4568ff,
            #7c4dff
        );

    color: white;

    border: 1px solid rgba(255,255,255,0.15);

    border-radius: 12px;

    font-weight: 600;

    padding: 10px;

    box-shadow:
        0 8px 25px rgba(80,80,255,0.25);

    transition: all 0.3s ease;
}


.stButton > button:hover {

    transform:
        translateY(-2px)
        scale(1.02);

    box-shadow:
        0 12px 35px rgba(100,80,255,0.45);

    border:
        1px solid rgba(255,255,255,0.35);
}


/* ---------- INFO BOXES ---------- */

[data-testid="stAlert"] {

    background:
        linear-gradient(
            135deg,
            rgba(50,100,255,0.12),
            rgba(130,70,255,0.08)
        );

    border:
        1px solid rgba(100,150,255,0.20);

    border-radius: 15px;

    backdrop-filter: blur(15px);
}


/* ---------- DIVIDER ---------- */

hr {

    border: none;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(120,140,255,0.35),
            transparent
        );
}


/* ---------- SCROLLBAR ---------- */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #050816;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            #557cff,
            #8b5cf6
        );

    border-radius: 20px;
}


/* ---------- FLOATING BACKGROUND GLOW ---------- */

.stApp::before {

    content: "";

    position: fixed;

    width: 450px;
    height: 450px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(60,100,255,0.18),
            transparent 70%
        );

    top: -150px;
    right: -120px;

    animation:
        floatingGlow 9s ease-in-out infinite;

    pointer-events: none;
}


.stApp::after {

    content: "";

    position: fixed;

    width: 400px;
    height: 400px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(130,70,255,0.14),
            transparent 70%
        );

    bottom: -150px;
    left: -100px;

    animation:
        floatingGlow 11s ease-in-out infinite reverse;

    pointer-events: none;
}


@keyframes floatingGlow {

    0% {
        transform: translate(0px, 0px);
    }

    50% {
        transform: translate(40px, 30px);
    }

    100% {
        transform: translate(0px, 0px);
    }
}


/* ---------- MOBILE ---------- */

@media (max-width: 768px) {

    .main-title {
        font-size: 35px;
    }

    .subtitle {
        font-size: 14px;
    }
}

</style>
""", unsafe_allow_html=True)
#============================================================================
#Create Rag system
#===========================================================================
@st.cache_resource
def create_rag_system():

    pdf_path = "data/attention.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

# print(f"Total pages loaded: {len(documents)}")
#
# print("\nFirst page:")
# print(documents[0].page_content[:1000])
#
# print("\nMetadata:")
# print(documents[0].metadata)
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)

# print(f"Original PDF pages: {len(documents)}")
# print(f"Total chunks created: {len(chunks)}")
#
# print("\nFirst Chunk:")
# print(chunks[0].page_content)
#
# print("\nMetadata:")
# print(chunks[0].metadata)

# 1. Initialize embedding models
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
#2. Create ChromaDB vector store
    persist_directory = "./attention_chroma_db"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="attention_paper"
    )
# print(
#     f"Total vectors stored:"
#     f"{vectorstore._collection.count()}"
# )
# Convert Vector Store into a Retriever

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
# Test the retriever
# query = "what is multi-head attention"
# retrieved_docs = retriever.invoke(query)
#
# print(f"\nRetrieved {len(retrieved_docs)} documents")

# for i, doc in enumerate(retrieved_docs, start=1):
#     print(f"\n========== Result {i} ==========")
#     print(doc.page_content)
#     print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# Create prompt
    system_prompt = """
    You are a question-answering assistant.
    
    Answer the user's question using ONLY the provided context
    retrieved from the research paper.
    
    If the answer is not available in the context, say:
    "I don't know based on the provided document."
    
    Keep the answer clear and concise.
    
    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

# --------------------------------------------------------------------------
# Create the document chain
# ---------------------------------------------------------------------------

# Combine retrieved documents with the prompt
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )
#-------------------------------------------------------------------------------------
# Create complete RAG chain
#----------------------------------------------------------------------------------
    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain, len(documents),len(chunks)

#  ===============================================================================
# LOAD RAG SYSTEM
# ===========================================================================
with st.spinner("Initializing research intelligence...."):
    rag_chain, total_pages, total_chunks=create_rag_system()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Research System")

    st.caption(
        "Retrieval-Augmented Generation"
    )

    st.divider()

    st.subheader("📄 Research Paper")

    st.write(
        "**Attention Is All You Need**"
    )

    st.write(
        f"📑 Pages processed: **{total_pages}**"
    )

    st.write(
        f"🧩 Knowledge chunks: **{total_chunks}**"
    )

    st.divider()

    st.subheader("🧠 AI Architecture")

    st.write("**LLM**")
    st.caption("Llama 3.3 70B")

    st.write("**Embedding Model**")
    st.caption("all-MiniLM-L6-v2")

    st.write("**Vector Database**")
    st.caption("ChromaDB")

    st.write("**Retrieval**")
    st.caption("Top 3 relevant chunks")

    st.divider()

    st.info(
        "The assistant answers questions using context "
        "retrieved directly from the research paper."
    )

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div style="text-align:center; margin-bottom:40px;">
<div class="research-badge">✦ AI-POWERED RESEARCH INTELLIGENCE</div>
<div class="main-title" style="margin-top:18px;">Attention Intelligence</div>
<div class="subtitle">Explore the architecture behind Transformers through retrieval-augmented intelligence.</div>
</div>
""",
    unsafe_allow_html=True
)
# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",

            "content":
                "Welcome to **Attention Intelligence**. 🧠\n\n"
                "Ask me anything about the "
                "**Attention Is All You Need** research paper."
        }

    ]


# ============================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about attention, transformers, encoders..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    # Add question to chat history

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # Display question

    with st.chat_message("user"):

        st.markdown(question)


    # Generate AI response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the research paper..."
        ):

            try:

                response = rag_chain.invoke({

                    "input": question

                })


                # Get final answer

                answer = response["answer"]


                # Display answer

                st.markdown(answer)


                # Add answer to chat history

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer

                })


            except Exception as e:

                st.error(
                    f"Unable to generate response: {str(e)}"
                )

# # Ask question
# print("\n===== Attention Is All You Need RAG Assistant =====")
#
# while True:
#     question = input("\nAsk your Question:")
#
#     if question.lower() == "exit":
#         print("RAG Assistant stopped.")
#         break
#
#     response = rag_chain.invoke({
#         "input": question
#     })
#
#     print("\nAnswer:")
#     print(response["answer"])
#
#
# # # Print only final answer
# # print("\n========== FINAL ANSWER ==========")
# # print(response["answer"])


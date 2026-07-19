## Hugging face
from langchain_huggingface import HuggingFaceEmbeddings

## initialize a simple Embeddign model(no api needed)
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# print(embeddings)

## create your first embeddings
text="Hello, I am learning about embeddings"

# embedding=embeddings.embed_query(text)
# print(f"Text: {text}")
# print(f"embedding length : {len(embedding)}")
# print(embedding)

sentences =[
    "The cat sat on the mat",
    "The dog played in the yard",
    "I love programming in Python",
    "Python is my favorite programming language"
]

embedding_sentence=embeddings.embed_documents(sentences)

print(embedding_sentence[0])
print(embedding_sentence[1])

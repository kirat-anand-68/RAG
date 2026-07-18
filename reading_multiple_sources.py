## create a simple text file
import os
os.makedirs("data/text_files",exist_ok=True)

sample_texts = {
    "data/text_files/python_intro.txt": """Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",

    "data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems


    """

}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8")as f:
        f.write(content)

print("Sample files be created")
## Text loader to read the single file

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

loader= TextLoader("data/text_files/python_intro.txt",encoding="utf-8")
# documents=loader.load()
# print(type(documents))
# print(documents)
single_documents=loader.load()
print("----- Single File Loader -----")
print(f"Loaded {len(single_documents)} document(s)")
print(f"Source : {single_documents[0].metadata['source']}")
print(f"Length : {len(single_documents[0].page_content)} characters")
print(f"Preview:\n{single_documents[0].page_content[:100]}...\n")

from langchain_community.document_loaders import DirectoryLoader

#load all the text files from the directory
dir_loader=DirectoryLoader(
    "data/text_files",## pattern to match files
    loader_cls=TextLoader, ## loader class to use
    loader_kwargs={'encoding':'utf-8'},
    show_progress=True
)
all_documents=dir_loader.load()

print("\n----- Directory Loader -----")
print(f"Loaded {len(all_documents)} document(s)\n")

for i, doc in enumerate(all_documents):
    print(f"Document {i+1}")
    print(f"Source : {doc.metadata['source']}")
    print(f"Length : {len(doc.page_content)} characters")
    print(f"Preview : {doc.page_content[:60]}...")
    print("-" * 50)

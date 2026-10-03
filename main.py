
import os

from langchain_openai import OpenAIEmbeddings

from file_manager import ReadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
file= ReadFile(path='links (1).txt')
documents = file.read_links()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents=documents)
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ("API_KEY"),
    check_embedding_ctx_length=False,
)
# embeddings = HuggingFaceEmbeddings(
#     model="Qwen/Qwen3-Embedding-0.6B",

# )
# vector = embeddings.embed_query(
#     "Django is a Python web framework."
# )

# print("Embedding size:", len(vector))
# print(vector[:5])
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
# print(len(vectorstore))
print(vectorstore)
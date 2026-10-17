
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from file_manager import ReadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv('.env_develpoment')

COLLECTION = 'documents'
file= ReadFile(path='links (1).txt')
pdf_paths=file.download_pdf()
print(pdf_paths)
documents=[]

for pdf in pdf_paths:
 
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    for doc in docs:
        doc.page_content = file.clean_docs(doc.page_content)
    documents.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents=documents)
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["API_KEY"],
    check_embedding_ctx_length=False,
)

if Path("./chroma_db").exists():
    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory="./chroma_db",
    )
vector_store.as_retriever()
data = vector_store._collection.get()


# embeddings = HuggingFaceEmbeddings(
#     model="Qwen/Qwen3-Embedding-0.6B",

# )
# vector = embeddings.embed_query(
#     "Django is a Python web framework."
# )

# print("Embedding size:", len(vector))
# print(vector[:5])

# # print(len(vectorstore))
# print(vectorstore)
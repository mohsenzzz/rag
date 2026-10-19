
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import PromptTemplate
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

prompt = PromptTemplate(
    template="""
    You are a helpful assistant answering questions based on the provided context.
    
    Rules:
    - Answer the question using only provided context.
    - If answer is not in the context, say o don't know.
    - Do not invent facts.
    - Give a clear and concise answer.
    
    Context:
        {context}
    
    Question:
    {question}
    
    Answer:
    """,
    input_variables=['context','question']
    
)

retriever= vector_store.as_retriever(
    search_kwargs={'k':3}
)

question = "این داکیومنت درباره چیست؟"

# Retrieve relevant chunks
docs = retriever.invoke(question)

# Combine the chunks into a single context string
context = "\n\n".join(
    doc.page_content for doc in docs
)

# Fill the prompt variables
formatted_prompt = prompt.invoke({
    "context": context,
    "question": question,
})

llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["API_KEY"],
)

response = llm.invoke(formatted_prompt.to_string())

print(response.content)
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
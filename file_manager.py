
import re

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

class ReadFile:
    def __init__(self,path:str):
        with open(path, 'r') as f:
            self.urls=f.readlines()
            
    def clean_docs(self, doc):
        soup = BeautifulSoup(doc.page_content, "html.parser")
        doc.page_content = soup.get_text(
                separator=" ",
                strip=True
            )
        return doc
    def read_links(self):
        documents= []
        for url in self.urls:
            loader = WebBaseLoader(url)
            doc =self.clean_docs(loader.load()[0])
            text = doc.page_content
            text = re.sub(r'\s+', ' ', text).strip()
            doc.page_content = text
            documents.append(doc)
        
        return documents
            
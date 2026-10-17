
from pathlib import Path
import re
import tempfile
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader



class ReadFile:
    def __init__(self,path:str):
        BASE_DIR = Path(__file__).resolve().parent
        self.DOWNLOAD_DIR = BASE_DIR / "downloads"

        self.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, 'r') as f:
            self.urls=f.readlines()
            
    def clean_docs(self, text):
        #Fix hyphenation across line breaks: exam-\nple -> example
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Replace multiple spaces and tabs with a single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces at the beginning and end of lines
        text = re.sub(r" *\n *", "\n", text)

        # Replace 3+ consecutive newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
    def download_pdf(self):
        pdf_paths =[]
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*;q=0.8",
        }
        with requests.Session() as session:
            session.headers.update(headers)
            for index, url in enumerate(self.urls):
                pdf_path = self.DOWNLOAD_DIR / f"document_{index}.pdf"
            
                    
                url = url.strip()
                response = session.get(
                    url,
                    timeout=30,
                    allow_redirects=True
                )
            
                pdf_path.write_bytes(response.content)
                pdf_paths.append(pdf_path)
        return pdf_paths
                

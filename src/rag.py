"""FAQ Retriever using ChromaDB and HuggingFace embeddings."""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class FAQRetriever:
    """Retrieves relevant FAQ content based on user questions."""

    def __init__(self, category: str = "general"):
        self.category = category.lower()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.category == "billing":
            pdf_file = "billing_faq.pdf"
            db_folder = "vectorstore_billing"
        else:
            pdf_file = "general_issues_faq.pdf"
            db_folder = "vectorstore_general"
        self.pdf_path = os.path.join(project_root, "data", pdf_file)
        self.database_path = os.path.join(project_root, "data", db_folder)
        print("Loading AI model to understand text...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.load_or_create_database()

    def load_or_create_database(self):
        if os.path.exists(self.database_path) and os.listdir(self.database_path):
            print(f"Loading {self.category} FAQ database...")
            self.vectorstore = Chroma(
                persist_directory=self.database_path,
                embedding_function=self.embeddings,
            )
            print("Database loaded!")
        else:
            print(f"Creating new {self.category} FAQ database...")
            self.create_database_from_pdf()

    def create_database_from_pdf(self):
        print(f"Reading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} searchable chunks")
        print("Building searchable database...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.database_path,
        )
        print("Database created and saved!")

    def search(self, question: str, num_results: int = 3) -> List[Document]:
        results = self.vectorstore.similarity_search(question, k=num_results)
        return results

    def format_results(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant information found in FAQ."
        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[FAQ Section {i}]\n{doc.page_content}")
        return "\n\n".join(formatted)


_cached_retrievers = {}


def get_faq_retriever(category: str = "general") -> FAQRetriever:
    global _cached_retrievers
    category = category.lower()
    if category not in _cached_retrievers:
        _cached_retrievers[category] = FAQRetriever(category=category)
    return _cached_retrievers[category]

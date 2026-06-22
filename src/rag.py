"""FAQ Retriever using ChromaDB and HuggingFace embeddings."""

import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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

        logger.info("Loading embedding model for %s retriever...", self.category)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model for {self.category} retriever: {exc}"
            ) from exc

        self.vectorstore = None
        self.load_or_create_database()

    def load_or_create_database(self):
        """Load an existing ChromaDB vectorstore or create one from the FAQ PDF."""
        if os.path.exists(self.database_path) and os.listdir(self.database_path):
            logger.info("Loading %s FAQ database from %s...", self.category, self.database_path)
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.database_path,
                    embedding_function=self.embeddings,
                )
                logger.info("%s database loaded.", self.category.capitalize())
            except Exception as exc:
                logger.warning(
                    "Failed to load existing %s database, rebuilding: %s",
                    self.category, exc,
                )
                self.create_database_from_pdf()
        else:
            logger.info("Creating new %s FAQ database...", self.category)
            self.create_database_from_pdf()

    def create_database_from_pdf(self):
        """Build a ChromaDB vectorstore from the category's FAQ PDF."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(
                f"FAQ PDF not found at {self.pdf_path}. "
                f"Please add the PDF to the data/ directory."
            )

        logger.info("Reading PDF: %s", self.pdf_path)
        try:
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read PDF {self.pdf_path}: {exc}"
            ) from exc

        if not documents:
            raise ValueError(
                f"PDF {self.pdf_path} produced no pages. "
                f"The file may be empty or corrupted."
            )

        logger.info("Loaded %d pages from %s", len(documents), self.pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError(
                f"PDF {self.pdf_path} produced no text chunks after splitting."
            )
        logger.info("Split into %d searchable chunks", len(chunks))

        logger.info("Building searchable database...")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.database_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create {self.category} vector database: {exc}"
            ) from exc
        logger.info("%s database created and saved.", self.category.capitalize())

    def search(self, question: str, num_results: int = 3) -> List[Document]:
        """Search the vectorstore for documents relevant to the question."""
        if self.vectorstore is None:
            raise RuntimeError(
                f"The {self.category} FAQ database is not initialized. "
                f"Cannot perform search."
            )
        try:
            return self.vectorstore.similarity_search(question, k=num_results)
        except Exception as exc:
            raise RuntimeError(
                f"Search failed in {self.category} FAQ database: {exc}"
            ) from exc

    def format_results(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant information found in FAQ."
        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[FAQ Section {i}]\n{doc.page_content}")
        return "\n\n".join(formatted)


_cached_retrievers = {}


def get_faq_retriever(category: str = "general") -> FAQRetriever:
    """Return a cached FAQRetriever for the given category.

    Raises:
        RuntimeError: If the retriever cannot be created.
    """
    global _cached_retrievers
    category = category.lower()
    if category not in _cached_retrievers:
        try:
            _cached_retrievers[category] = FAQRetriever(category=category)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize {category} FAQ retriever: {exc}"
            ) from exc
    return _cached_retrievers[category]

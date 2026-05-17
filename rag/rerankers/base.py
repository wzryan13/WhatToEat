# rag/rerankers/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseReranker(ABC):
    """
    Abstract base class for all reranker implementations.
    """
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks and/or filters a list of documents based on a query.

        Args:
            query: The user's query.
            documents: The list of documents to rerank.

        Returns:
            A new list of documents, filtered and/or sorted by relevance.
        """
        pass

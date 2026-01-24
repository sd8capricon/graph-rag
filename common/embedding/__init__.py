"""Singleton embedding service for project-wide usage."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class EmbeddingService:
    """Singleton service for managing embedding model instance across the application."""

    _instance: Optional[EmbeddingService] = None
    _embedding: Optional[GoogleGenerativeAIEmbeddings] = None

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding(self) -> GoogleGenerativeAIEmbeddings:
        """Get or initialize the embedding model instance.

        Returns:
            GoogleGenerativeAIEmbeddings: The singleton embedding model instance.
        """
        if self._embedding is None:
            self._embedding = GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
                output_dimensionality=int(os.getenv("EMBEDDING_DIMENSIONALITY", "768")),
            )
        return self._embedding

    def reset(self) -> None:
        """Reset the embedding model instance (useful for testing or reconfiguration)."""
        self._embedding = None


def get_embedding() -> GoogleGenerativeAIEmbeddings:
    """Get the singleton embedding model instance.

    Returns:
        GoogleGenerativeAIEmbeddings: The singleton embedding model instance.
    """
    service = EmbeddingService()
    return service.get_embedding()

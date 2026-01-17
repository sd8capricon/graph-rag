"""Singleton LLM service for project-wide usage."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class LLMService:
    """Singleton service for managing LLM instance across the application."""

    _instance: Optional[LLMService] = None
    _llm: Optional[ChatOpenAI] = None

    def __new__(cls) -> LLMService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self) -> ChatOpenAI:
        """Get or initialize the LLM instance.

        Returns:
            ChatOpenAI: The singleton LLM instance.
        """
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free"),
                base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                reasoning_effort="medium",
            )
        return self._llm

    def reset(self) -> None:
        """Reset the LLM instance (useful for testing or reconfiguration)."""
        self._llm = None


def get_llm() -> ChatOpenAI:
    """Get the singleton LLM instance.

    Returns:
        ChatOpenAI: The singleton LLM instance.
    """
    service = LLMService()
    return service.get_llm()

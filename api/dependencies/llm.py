from langchain_openai import ChatOpenAI

from common.llm import LLMService


def get_llm() -> ChatOpenAI:
    """Get the singleton LLM instance.

    Returns:
        ChatOpenAI: The singleton LLM instance.
    """
    service = LLMService()
    return service.get_llm()

from common.services.knowledge_base import KnowledgeBaseService

_knowledge_base_service: KnowledgeBaseService | None = None


def set_kb_service(service: KnowledgeBaseService):
    global _knowledge_base_service
    _knowledge_base_service = service


def get_kb_service() -> KnowledgeBaseService:
    if not _knowledge_base_service:
        raise ValueError("Knowledge Base service not initialized")
    return _knowledge_base_service

from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    thread_id: str | None = Field(None, description="Conversation thread id")
    query: str = Field(..., description="User input message")


class ChatResponse(BaseModel):
    thread_id: str = Field(..., description="Conversation thread id")
    message: Any = Field(..., description="Assistant message")

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="User input message")

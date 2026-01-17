from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    body: str = Field(
        description="Detailed answer to the user's query based on context."
    )
    follow_up_questions: list[str] = Field(
        description="Follow up questions based on context"
    )


class Node(BaseModel):
    query: str
    answer: str
    # Use a default_factory via Field to ensure each instance gets a fresh list
    children: list[Node] = Field(default_factory=list)

    def add_child(self, node: Node):
        self.children.append(node)

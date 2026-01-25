from fastapi import FastAPI

from api.lifespan import lifespan
from api.routers import chat, knowledge_base

app = FastAPI(lifespan=lifespan, root_path="/v1")


@app.get("/")
async def index():
    return {"message": "GraphRAG API"}


app.include_router(chat.router, prefix="/chat")
app.include_router(knowledge_base.router, prefix="/knowledge_base")

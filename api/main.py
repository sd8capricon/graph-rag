from fastapi import FastAPI

from api.lifespan import lifespan
from api.routers import chat

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return {"message": "GraphRAG API"}


app.include_router(chat.router, prefix="/chat")

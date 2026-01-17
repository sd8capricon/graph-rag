from fastapi import FastAPI
from api.routers import chat

app = FastAPI()


@app.get("/")
async def index():
    return {"message": "GraphRAG API"}


app.include_router(chat.router, prefix="/chat")

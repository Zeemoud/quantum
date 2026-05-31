"""
Quantum API — FastAPI entry point.
Run with: uvicorn api.server:app --reload
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

load_dotenv()

app = FastAPI(
    title="Quantum API",
    description="REST API for the Quantum language model",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "quantum"}

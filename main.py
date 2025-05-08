from fastapi import FastAPI
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="PDF Topic Extractor",
    version="1.0"
)

origins = [
    "http://localhost",
    "http://localhost:5174",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

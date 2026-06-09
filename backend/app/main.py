from fastapi import FastAPI

from app.shared.config import settings
from app.shared.database import Base, engine

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
    }
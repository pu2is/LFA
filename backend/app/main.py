from fastapi import FastAPI

from app.modules.files.routes import router as paths_router
from app.modules.labeling.routes import router as labels_router
from app.modules.processing.routes import router as processing_router
from app.modules.scans.routes import router as scans_router
from app.shared.config import settings
from app.shared.database import Base, engine

app = FastAPI(title=settings.app_name)

app.include_router(paths_router)
app.include_router(labels_router)
app.include_router(scans_router)
app.include_router(processing_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
    }
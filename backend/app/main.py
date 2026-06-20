from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.modules.events.routes import router as events_router
from app.modules.files.routes import router as paths_router
from app.modules.labeling.routes import router as labels_router
from app.modules.processing.routes import router as processing_router
from app.modules.scans.routes import router as scans_router
from app.shared.config import settings
from app.shared.database import Base, engine, get_db

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paths_router)
app.include_router(labels_router)
app.include_router(scans_router)
app.include_router(processing_router)
app.include_router(events_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.delete("/debug/reset")
def debug_reset_db(db: Session = Depends(get_db)):
    """DEV ONLY: truncate all tables. Respects FK order via CASCADE."""
    from sqlalchemy import text

    for table in reversed(Base.metadata.sorted_tables):
        db.execute(text(f"TRUNCATE TABLE {table.name} CASCADE"))
    db.commit()
    return {"status": "ok", "message": "All tables cleared"}

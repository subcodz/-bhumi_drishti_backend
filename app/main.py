from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text

from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base
from app.db.models import RoadSegment, SegmentFeature, SegmentRisk, HistoricalIncident, FieldReport  # noqa: F401
from app.api.segments import router as segments_router
from app.api.features import router as features_router
from app.api.reports import router as reports_router
from app.api.ml import router as ml_router
from app.api.dashboard import router as dashboard_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure PostGIS extension and tables exist on startup if database is accessible
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[Warning] Database connection deferred at startup: {e}")
    yield


app = FastAPI(
    title="Bhumi Drishti — Road Risk & Accessibility Backend Prototype",
    description="GIS Road Risk & Accessibility backend with feature vectors, IMD weather sync, field reporting, & ML prediction pipeline for North Eastern Region (NER) of India",
    version="0.6.0",
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard (helper_dashboard / React / Vite / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


app.include_router(segments_router, prefix="/api/v1")
app.include_router(features_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")

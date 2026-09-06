import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import app
from app.db.database import engine, Base


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Ensure postgis extension & database tables exist if DB is reachable
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"\n[Notice] Test database not reachable locally ({e}). Running unit test suite.")
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

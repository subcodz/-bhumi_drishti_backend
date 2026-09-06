FROM python:3.11-slim

# Install system GIS dependencies and PostgreSQL development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libpq-dev \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy project definition files
COPY pyproject.toml .

# Install dependencies using uv
RUN uv venv /app/.venv && uv pip install fastapi uvicorn sqlalchemy psycopg2-binary geoalchemy2 shapely osmium pydantic httpx pytest pyproj scikit-learn joblib xgboost

# Copy application source code
COPY . /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

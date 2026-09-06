#!/bin/sh
set -e

echo "================================================================="
echo " STARTING ROAD RISK & ACCESSIBILITY BACKEND CONTAINER (NER INDIA)"
echo "================================================================="

PORT="${PORT:-8000}"

# Run database setup & initial seeding in background so web server binds port immediately
if [ "$RUN_POPULATE_ON_STARTUP" = "true" ]; then
    echo "Starting automated database initialization in background..."
    (
        sleep 2
        python scripts/populate_db.py || echo "Warning: Background database setup encountered an issue."
        echo "Background database initialization complete!"
    ) &
fi

echo "Binding and starting FastAPI application on port ${PORT}..."
if [ $# -gt 0 ] && [ "$1" != "uvicorn" ]; then
    exec "$@"
else
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
fi

from __future__ import annotations

"""WSGI entry for Gunicorn."""

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from .app import app as dash_app
from .api import router as api_router # Import the API router

server = FastAPI(title="Omesh Dashboard API & App") # Added title

# Mount the API router under /api
server.include_router(api_router)

# Mount the Dash app at the root
# IMPORTANT: This should typically be the last mount if it's at the root.
server.mount("/", WSGIMiddleware(dash_app.server))

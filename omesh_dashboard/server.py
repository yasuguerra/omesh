from __future__ import annotations

"""WSGI entry for Gunicorn."""

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from .app import app as dash_app

server = FastAPI()
server.mount("/", WSGIMiddleware(dash_app.server))

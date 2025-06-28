"""Central settings loader for Omesh Super-Dashboard."""
from __future__ import annotations

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env once
_env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=_env_path, override=False)

DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = int(os.getenv('DB_PORT', '3306')) if os.getenv('DB_PORT') else None
DB_USER = os.getenv('DB_USER', '')
DB_PASSWD = os.getenv('DB_PASSWD', '')
DB_DATABASE = os.getenv('DB_DATABASE', '')

GA_PROPERTY_ID = os.getenv('GA_PROPERTY_ID', '')
GA_KEY_PATH = os.getenv('GA_KEY_PATH', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

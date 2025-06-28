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

# Facebook Graph API (for Social media integration)
FB_ACCESS_TOKEN = os.getenv('FB_ACCESS_TOKEN', '')
# Note: For Instagram, Facebook Page needs to be connected to Instagram Professional Account.
# The FB_ACCESS_TOKEN is often a Page Access Token with ads_management, business_management,
# pages_show_list, pages_read_engagement, pages_read_user_content, instagram_basic, instagram_manage_insights.
# Also need FACEBOOK_ID and INSTAGRAM_ID for social.py, these are not secret.
# For now, these IDs are hardcoded in social.py but could be moved to .env if they change often.
FACEBOOK_ID = os.getenv('FACEBOOK_ID', 'your_facebook_page_id') # Example, if you want it from .env
INSTAGRAM_ID = os.getenv('INSTAGRAM_ID', 'your_instagram_business_id') # Example

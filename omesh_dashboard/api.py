from __future__ import annotations

"""
FastAPI Endpoints for Omesh Dashboard.

Exposes Plotly figures as JSON for consumption by other services
or a future React/Next.js frontend.
"""
import logging
from fastapi import APIRouter, Query, HTTPException, Depends
import plotly.io as pio
import pandas as pd
from datetime date, timedelta

# Import data access functions
from .data import ga_data_access as ga_data
from .data import google_ads_api as ads_data_api # For Google Ads
from .data.data_processing import get_facebook_posts, get_instagram_posts, process_facebook_posts, process_instagram_posts # For Social
# Sales data access would go here if it were from a DB, e.g.:
# from .data import data_access as sales_db_data

# Import figure generation functions from page modules
# Note: These get_fig functions expect DataFrames. The API endpoints will fetch data first.
from .pages.google_analytics import (
    get_fig_sessions_overview, get_fig_users_overview, get_fig_conversions_overview,
    get_fig_conversion_rate_overview, get_fig_normalized_trends_overview,
    # Add other GA figure functions as they are created/needed for API
)
from .pages.google_ads import get_fig_daily_performance_trends
from .pages.social import get_fig_instagram_impressions_trend, get_fig_instagram_engagement_by_type
# Sales figure functions are more complex due to CSV upload dependency.
# For API, they would typically rely on data fetched from a DB.
# from .pages.sales import get_fig_flights_by_month # Example

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Dashboard Figures"])

# --- Helper for Date Parameters ---
def get_date_params(
    start_date: date | None = Query(None, description="Start date (YYYY-MM-DD). Defaults to 30 days ago."),
    end_date: date | None = Query(None, description="End date (YYYY-MM-DD). Defaults to today.")
) -> tuple[str, str]:
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=29) # Default to 30 days window

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date cannot be after end date.")

    return start_date.isoformat(), end_date.isoformat()

# --- Google Analytics Endpoints ---
@router.get("/ga/traffic_overview/sessions")
async def ga_sessions_figure_json(dates: tuple[str, str] = Depends(get_date_params)):
    start_date_str, end_date_str = dates
    df_overview = ga_data.get_traffic_overview_df(start_date_str, end_date_str)
    if df_overview.empty:
        raise HTTPException(status_code=404, detail=f"No GA sessions data found for {start_date_str} to {end_date_str}")
    fig = get_fig_sessions_overview(df_overview)
    return {"figure": pio.to_json(fig)}

@router.get("/ga/traffic_overview/users")
async def ga_users_figure_json(dates: tuple[str, str] = Depends(get_date_params)):
    start_date_str, end_date_str = dates
    df_overview = ga_data.get_traffic_overview_df(start_date_str, end_date_str)
    if df_overview.empty:
        raise HTTPException(status_code=404, detail=f"No GA users data found for {start_date_str} to {end_date_str}")
    fig = get_fig_users_overview(df_overview)
    return {"figure": pio.to_json(fig)}

# Add more GA endpoints similarly...
# Example: Normalized Trends
@router.get("/ga/traffic_overview/normalized_trends")
async def ga_normalized_trends_figure_json(dates: tuple[str, str] = Depends(get_date_params)):
    start_date_str, end_date_str = dates
    df_overview = ga_data.get_traffic_overview_df(start_date_str, end_date_str) # Fetches sessions, activeUsers, conversions
    if df_overview.empty:
        raise HTTPException(status_code=404, detail=f"No GA data for normalized trends for {start_date_str} to {end_date_str}")
    fig = get_fig_normalized_trends_overview(df_overview) # This function handles the normalization
    return {"figure": pio.to_json(fig)}

# --- Google Ads Endpoints ---
@router.get("/ads/daily_performance_trends")
async def ads_daily_trends_figure_json(dates: tuple[str, str] = Depends(get_date_params)):
    start_date_iso, end_date_iso = dates
    # Google Ads API functions in `google_ads_api.py` expect datetime.date objects
    start_date_obj = date.fromisoformat(start_date_iso)
    end_date_obj = date.fromisoformat(end_date_iso)

    daily_df = ads_data_api.fetch_daily_performance(start_date_obj, end_date_obj)
    if daily_df.empty:
        raise HTTPException(status_code=404, detail=f"No Google Ads daily performance data found for {start_date_iso} to {end_date_iso}")
    fig = get_fig_daily_performance_trends(daily_df)
    return {"figure": pio.to_json(fig)}

# --- Social Media Endpoints ---
@router.get("/social/instagram_impressions_trend")
async def social_ig_impressions_figure_json(dates: tuple[str, str] = Depends(get_date_params)):
    start_date_str, end_date_str = dates
    # Fetch and process Instagram data
    # Note: INSTAGRAM_ID should be available from settings
    from .settings import INSTAGRAM_ID
    if not INSTAGRAM_ID:
         raise HTTPException(status_code=500, detail="INSTAGRAM_ID not configured in settings.")

    ig_posts_raw = get_instagram_posts(INSTAGRAM_ID)
    df_ig_full = process_instagram_posts(ig_posts_raw)

    df_ig = df_ig_full.copy()
    if not df_ig.empty and 'timestamp' in df_ig.columns:
        df_ig['timestamp'] = pd.to_datetime(df_ig['timestamp'], errors='coerce').dt.tz_localize(None)
        df_ig = df_ig.dropna(subset=['timestamp'])
        # Filter by date
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        df_ig = df_ig[(df_ig['timestamp'] >= start_dt) & (df_ig['timestamp'] <= end_dt)]

    if df_ig.empty:
        raise HTTPException(status_code=404, detail=f"No Instagram impression data found for {start_date_str} to {end_date_str}")

    fig = get_fig_instagram_impressions_trend(df_ig)
    return {"figure": pio.to_json(fig)}

# --- Sales Endpoints (Placeholder/Example) ---
# Sales data currently comes from CSV uploads in the Dash app.
# For a true API, sales data would ideally be in a database.
# This is a placeholder showing how it might work if data was queryable.
@router.get("/sales/flights_by_month_example")
async def sales_flights_by_month_figure_json(
    # Example: these filters would query a DB
    year: int = Query(date.today().year, description="Year to filter sales data"),
    destination_filter: str | None = Query(None, description="Optional filter by destination")
):
    # In a real scenario:
    # 1. Fetch sales data from DB based on year, destination_filter, etc.
    #    df_sales = sales_db_data.get_sales_records(year=year, destination=destination_filter)
    # 2. Process df_sales as needed (e.g., ensure 'MonthName', 'Año' columns)
    #    df_sales['MonthName'] = pd.to_datetime(df_sales['date_column']).dt.strftime('%B') # Example
    #    df_sales['Año'] = pd.to_datetime(df_sales['date_column']).dt.year # Example
    # For now, creating a mock DataFrame structure that get_fig_flights_by_month expects
    logger.warning("Sales API endpoint is using placeholder data. Implement database connection for real data.")
    mock_data = {
        'Año': [year, year, year-1, year-1],
        'MonthName': ['January', 'February', 'January', 'February'],
        # This is just to make px.line work, actual data would be counts from .size()
        'Flights_count_placeholder': [10,15, 8, 12]
    }
    df_mock_sales_agg = pd.DataFrame(mock_data)
    # The actual get_fig_flights_by_month expects a DataFrame that it will then groupby.
    # This is a simplified example.
    # from .pages.sales import get_fig_flights_by_month
    # fig = get_fig_flights_by_month(df_mock_sales_agg) # This won't work directly as get_fig expects raw data to group

    # Let's simulate the grouped data the figure function expects
    # flights_data = df_plot.groupby(['Año', 'MonthName'], observed=False).size().reset_index(name='Flights')
    # For the mock, we'll just use the placeholder counts as 'Flights'
    df_mock_sales_grouped = df_mock_sales_agg.rename(columns={'Flights_count_placeholder':'Flights'})

    # Since I cannot import get_fig_flights_by_month from sales due to its complexity and potential circular deps for now,
    # I'll create a simplified version of the figure here for demonstration.
    if df_mock_sales_grouped.empty:
        raise HTTPException(status_code=404, detail="No mock sales data available.")

    fig = px.line(df_mock_sales_grouped, x='MonthName', y='Flights', color='Año', markers=True,
                  title=f'Flights by Month (Example for {year})')
    fig.update_layout(margin=FIGURE_MARGIN, xaxis_title="Month", yaxis_title="Number of Flights")

    return {"figure": pio.to_json(fig), "detail": "This is placeholder data for sales."}

# TODO: Add more endpoints for other figures from all pages.
# Each endpoint should:
# 1. Define necessary path/query parameters (dates, filters).
# 2. Call the appropriate data access function(s).
# 3. Call the corresponding get_fig_{name} function from the page module.
# 4. Return the Plotly JSON.
# 5. Handle errors gracefully (e.g., return 404 if no data, 500 for server errors).
```

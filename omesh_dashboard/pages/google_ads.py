############################################
# Omesh Dashboard – Google Ads Module
############################################
"""
Google Ads Overview Page.

Shows key performance indicators (KPIs), daily performance trends,
and a detailed campaign performance table.
Data is fetched via the `google_ads_api.py` module.
"""

from __future__ import annotations

import datetime as _dt
import logging # Added logging
import typing # Added typing

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Added for go.Figure typing
from dash import Input, Output, State, dcc, html, callback # Added callback decorator explicitly

from ..data import google_ads_api as gads

logger = logging.getLogger(__name__)

# Standard margin for all figures
FIGURE_MARGIN = dict(t=40, l=20, r=20, b=40)
DEFAULT_NO_DATA_FIGURE_TITLE = "No data to display"

# Helper function to create an empty figure
def create_empty_figure(title: str = DEFAULT_NO_DATA_FIGURE_TITLE) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': title, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}],
        margin=FIGURE_MARGIN
    )
    return fig

##############################################################################
# 🔧  UI Helper Components
##############################################################################

def _kpi_card(title: str, value: str, delta: float | str, icon_class: str = "") -> dbc.Card:
    """Creates a KPI card component."""
    try:
        delta_val = float(delta)
        color = "success" if delta_val >= 0 else "danger"
        prefix = "+" if delta_val >= 0 else ""
        delta_str = f"{prefix}{delta_val:.1f}%"
        progress_val = min(abs(delta_val), 100)
    except ValueError: # Handle cases where delta might be "N/A" or similar
        color = "secondary"
        delta_str = str(delta)
        progress_val = 0

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(title, className="text-muted small me-1"),
                html.I(className=f"{icon_class} text-muted") if icon_class else None,
            ], className="d-flex align-items-center"),
            html.H3(value, className="mb-0 fw-bold"),
            html.Small(f"{delta_str} vs prev. period", className=f"text-{color}"),
            dbc.Progress(value=progress_val, color=color if color != "secondary" else "dark", className="mt-2", style={"height": "6px"}),
        ]), className="shadow-sm rounded-3 h-100", # Added h-100 for consistent height if in a row
    )


def _date_picker_range() -> dcc.DatePickerRange:
    """Creates the date picker range component for Google Ads."""
    today = _dt.date.today()
    return dcc.DatePickerRange(
        id="gads-date-range-picker", # Specific ID
        min_date_allowed=today - _dt.timedelta(days=365 * 2), # Allow up to 2 years back
        max_date_allowed=today,
        initial_visible_month=today,
        start_date=today - _dt.timedelta(days=29), # Default to last 30 days (29 + today)
        end_date=today,
        display_format="YYYY-MM-DD",
        className="me-2",
    )

##############################################################################
# 📊 Figure Generation Function
##############################################################################

def get_fig_daily_performance_trends(daily_df: pd.DataFrame) -> go.Figure:
    """Generates the daily performance trends bar chart."""
    if daily_df.empty:
        return create_empty_figure("Daily Performance Trends (No data)")

    # Ensure essential columns exist
    required_cols = ["date", "spend", "clicks", "conversions"]
    if not all(col in daily_df.columns for col in required_cols):
        logger.warning(f"Daily performance data missing one or more required columns: {required_cols}")
        return create_empty_figure("Daily Performance Trends (Missing data columns)")

    fig = px.bar(daily_df, x="date", y=["spend", "clicks", "conversions"], barmode="group",
                 labels={"value": "Metric Value", "variable": "Metric Name", "date": "Date"})
    fig.update_layout(
        legend_title_text="Metrics",
        xaxis_title="Date",
        yaxis_title="Value",
        margin=FIGURE_MARGIN, # Apply standard margin
        title="Daily Performance Trends"
    )
    return fig

##############################################################################
# 🎨  Layout Definition
##############################################################################

def layout(_app: dash.Dash | None = None) -> dbc.Container: # app argument is optional now
    """
    Returns the layout for the Google Ads Overview page.
    The `_app` argument is kept for compatibility with `app.py` but not strictly needed here.
    """
    return dbc.Container([
        dcc.Store(id="gads-overview-data-store"), # Specific ID

        # Controls Row
        dbc.Row([
            dbc.Col(_date_picker_range(), width="auto"),
            dbc.Col(dbc.Button("Refresh Data", id="gads-refresh-button", color="primary"), width="auto"), # Specific ID
            dbc.Col(html.Div(id="gads-last-updated-timestamp", className="small text-muted align-self-center"), width="auto"), # Specific ID
        ], className="gy-2 align-items-center my-3"), # Adjusted margin

        # Content Area (KPIs, Chart, Table)
        dcc.Loading(id="gads-loading-indicator", type="default", children=[
             html.Div(id="gads-main-content-area") # Specific ID
        ]),

    ], fluid=True, className="pt-3")

##############################################################################
# ⚡  Callbacks
##############################################################################

@callback( # Using dash.callback decorator
    Output("gads-overview-data-store", "data"),
    Output("gads-last-updated-timestamp", "children"),
    Input("gads-refresh-button", "n_clicks"),
    State("gads-date-range-picker", "start_date"),
    State("gads-date-range-picker", "end_date"),
    # prevent_initial_call=True # Can be true if we don't want an initial load without clicking refresh
)
def refresh_google_ads_data(n_clicks: int | None, start_date_str: str, end_date_str: str) -> tuple[dict | None, str]:
    """Fetches fresh data from Google Ads API based on selected date range."""
    # This callback will run on page load if prevent_initial_call is False or not set
    # and also when the refresh button is clicked.

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else "initial_load"

    logger.info(f"Refreshing Google Ads data. Trigger: {triggered_id}. Start: {start_date_str}, End: {end_date_str}")

    if not start_date_str or not end_date_str:
        return None, "Please select a valid date range."

    try:
        start_date = _dt.date.fromisoformat(start_date_str)
        end_date = _dt.date.fromisoformat(end_date_str)
    except ValueError:
        logger.error(f"Invalid date format received: Start: {start_date_str}, End: {end_date_str}")
        return None, "Error: Invalid date format."

    try:
        overview_data = gads.fetch_overview(start_date, end_date)
        daily_data_df = gads.fetch_daily_performance(start_date, end_date)
        campaigns_data_df = gads.fetch_campaign_performance(start_date, end_date)

        # Convert DataFrames to dict for storage only if they are not empty
        daily_data_dict = daily_data_df.to_dict("records") if not daily_data_df.empty else []
        campaigns_data_dict = campaigns_data_df.to_dict("records") if not campaigns_data_df.empty else []


        processed_data = {
            "overview": overview_data, # This is already a dict
            "daily": daily_data_dict,
            "campaigns": campaigns_data_dict,
        }
        timestamp_str = f"Last updated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return processed_data, timestamp_str
    except Exception as e:
        logger.error(f"Error fetching Google Ads data: {e}", exc_info=True)
        return None, f"Error fetching data: {str(e)}"


@callback( # Using dash.callback decorator
    Output("gads-main-content-area", "children"),
    Input("gads-overview-data-store", "data"),
    # prevent_initial_call=True # Content should render when data is available
)
def render_google_ads_content(stored_data: dict | None) -> list[dbc.Row | dcc.Graph | dash_table.DataTable | dbc.Alert]:
    """Renders the main content (KPIs, chart, table) from stored data."""
    if not stored_data:
        return [dbc.Alert("Click 'Refresh Data' to load Google Ads information for the selected period.", color="info", className="mt-3")]

    overview = stored_data.get("overview", {})
    daily_df = pd.DataFrame(stored_data.get("daily", []))
    campaigns_df = pd.DataFrame(stored_data.get("campaigns", []))

    # 1. KPI cards
    kpi_cards_row = dbc.Row([
        dbc.Col(_kpi_card("Spend", f"${overview.get('spend', 0):,.0f}", overview.get('delta_spend_pct', "N/A"), "bi bi-cash-coin"), md=6, lg=3, className="mb-3"),
        dbc.Col(_kpi_card("Clicks", f"{overview.get('clicks', 0):,}", overview.get('delta_clicks_pct', "N/A"), "bi bi-cursor-fill"), md=6, lg=3, className="mb-3"), # Changed icon
        dbc.Col(_kpi_card("Avg. CPC", f"${overview.get('cpc', 0):.2f}", overview.get('delta_cpc_pct', "N/A"), "bi bi-mouse2"), md=6, lg=3, className="mb-3"), # Changed icon
        dbc.Col(_kpi_card("ROAS", f"{overview.get('roas', 0):.1f}x", overview.get('delta_roas_pct', "N/A"), "bi bi-graph-up-arrow"), md=6, lg=3, className="mb-3"), # Changed icon
    ], className="g-3 mb-4") # g-3 for gutter

    # 2. Trends chart
    trends_chart_component = dcc.Graph(
        id="gads-daily-trends-chart", # Specific ID
        figure=get_fig_daily_performance_trends(daily_df),
        config={"displayModeBar": False}, # No Plotly mode bar
        className="mb-4 shadow-sm"
    )

    # 3. Campaign table
    campaign_table_component = html.Div([
        html.H4("Campaign Performance", className="mb-3"), # Added title for table
        dash_table.DataTable(
            id="gads-campaign-performance-table", # Specific ID
            data=campaigns_df.sort_values("spend", ascending=False).to_dict("records") if not campaigns_df.empty else [],
            columns=[{"name": col.replace("_", " ").title(), "id": col} for col in campaigns_df.columns] if not campaigns_df.empty else [],
            sort_action="native",
            filter_action="native",
            page_size=10, # Reduced page size for better view
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold", "backgroundColor": "rgb(230, 230, 230)"}, # Light grey header
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_data_conditional=[{ # Zebra stripes
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }]
        )
    ], className="mt-4") if not campaigns_df.empty else dbc.Alert("No campaign data available for this period.", color="light", className="mt-3")


    return [kpi_cards_row, trends_chart_component, campaign_table_component]


def register_callbacks(_app: dash.Dash) -> None: # app argument is optional
    """
    Callbacks are registered at import time using the @dash.callback decorator.
    This function is kept for structural consistency but is a no-op.
    """
    pass

# Note: The original `get_google_ads_tab` function was essentially the layout.
# It's now directly returned by the `layout()` function.
# The `app` argument in `layout()` and `register_callbacks()` is made optional
# as it's not strictly needed when using `@dash.callback` for registering callbacks.
# It's kept for potential compatibility if `app.py` calls it with an app instance.
```

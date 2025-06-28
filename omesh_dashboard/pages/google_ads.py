############################################
# SkyIntel – Google Ads Tab (single‑pane Overview)
############################################
"""
Refactored version: **all Google Ads insights live in ONE view** called “Overview”.
It shows:
  • KPI cards (spend, clicks, CPC, ROAS)
  • Daily trends chart (bar grouped)
  • Campaign performance table

The module still plugs directly into `google_ads_api.py` (real data; no mocks).
"""

from __future__ import annotations

import datetime as _dt

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html

from ..data import google_ads_api as gads  # 🔌 real integration

##############################################################################
# 🔧  Helpers
##############################################################################

def _kpi_card(title: str, value: str, delta: float | str, icon: str = "") -> dbc.Card:
    color = "success" if isinstance(delta, (int, float)) and delta >= 0 else "danger"
    prefix = "+" if isinstance(delta, (int, float)) and delta >= 0 else ""
    delta_str = f"{prefix}{delta:.1f}%" if isinstance(delta, (int, float)) else str(delta)
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(title, className="text-muted small me-1"),
                html.I(className=f"{icon} text-muted") if icon else None,
            ], className="d-flex align-items-center"),
            html.H3(value, className="mb-0 fw-bold"),
            html.Small(delta_str + " vs prev. period", className=f"text-{color}"),
            dbc.Progress(value=min(abs(delta) if isinstance(delta, (int, float)) else 0, 100),
                         color="dark", className="mt-2", style={"height": "6px"}),
        ]), className="shadow-sm rounded-3",
    )


def _date_picker() -> dcc.DatePickerRange:
    today = _dt.date.today()
    return dcc.DatePickerRange(
        id="gads-date-range",
        min_date_allowed=today - _dt.timedelta(days=365),
        max_date_allowed=today,
        start_date=today - _dt.timedelta(days=7),
        end_date=today,
        display_format="YYYY‑MM‑DD",
        className="me-2",
    )

##############################################################################
# 🎨  Layout factory
##############################################################################

def get_google_ads_tab(app: dash.Dash) -> dbc.Container:
    """Embeddable Google Ads overview pane."""
    return dbc.Container([
        dcc.Store(id="gads-overview-store"),

        # Controls
        dbc.Row([
            dbc.Col(_date_picker(), width="auto"),
            dbc.Col(dbc.Button("Refresh", id="gads-refresh", color="primary"), width="auto"),
            dbc.Col(html.Div(id="gads-last-updated", className="small text-muted"), width="auto"),
        ], className="gy-2 align-items-center my-2"),

        # Content populated via callback
        html.Div(id="gads-content"),
    ], fluid=True, className="pt-3")

##############################################################################
# ⚡  Callbacks
##############################################################################

@dash.callback(
    Output("gads-overview-store", "data"),
    Output("gads-last-updated", "children"),
    Input("gads-refresh", "n_clicks"),
    State("gads-date-range", "start_date"),
    State("gads-date-range", "end_date"),
    prevent_initial_call=True,
)
def _refresh_data(_, start_date, end_date):
    start = _dt.date.fromisoformat(start_date)
    end = _dt.date.fromisoformat(end_date)

    overview = gads.fetch_overview(start, end)
    daily = gads.fetch_daily_performance(start, end).to_dict("records")
    campaigns = gads.fetch_campaign_performance(start, end).to_dict("records")

    ts = _dt.datetime.now().strftime("Last updated %Y‑%m‑%d %H:%M:%S")
    return {"overview": overview, "daily": daily, "campaigns": campaigns}, ts


@dash.callback(
    Output("gads-content", "children"),
    Input("gads-overview-store", "data"),
    prevent_initial_call=True,
)
def _render_content(data):
    if not data:
        return dbc.Alert("Click Refresh to load Google Ads data.", color="info")

    # 1. KPI cards
    o = data["overview"]
    kpi_row = dbc.Row([
        dbc.Col(_kpi_card("Spend", f"${o['spend']:,}", o['delta_spend_pct'], "bi bi-cash-coin"), md=3),
        dbc.Col(_kpi_card("Clicks", f"{o['clicks']:,}", o['delta_clicks_pct'], "bi bi-cursor"), md=3),
        dbc.Col(_kpi_card("Avg. CPC", f"${o['cpc']:.2f}", o['delta_cpc_pct'], "bi bi-currency-dollar"), md=3),
        dbc.Col(_kpi_card("ROAS", f"{o['roas']:.1f}x", o['delta_roas_pct'], "bi bi-graph-up"), md=3),
    ], className="g-3 mb-4")

    # 2. Trends chart
    daily_df = pd.DataFrame(data["daily"])
    fig = px.bar(daily_df, x="date", y=["spend", "clicks", "conversions"], barmode="group",
                 labels={"value": "Metric", "variable": ""})
    fig.update_layout(legend_title="", xaxis_title="Date", yaxis_title="", margin=dict(t=10))
    chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, className="mb-4")

    # 3. Campaign table
    camp_df = pd.DataFrame(data["campaigns"]).sort_values("spend", ascending=False)
    table = dash.dash_table.DataTable(
        data=camp_df.to_dict("records"),
        columns=[{"name": c.capitalize(), "id": c} for c in camp_df.columns],
        sort_action="native",
        filter_action="native",
        page_size=15,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
    )

    return [kpi_row, chart, table]

##############################################################################
#   END OF MODULE – import get_google_ads_tab and embed it anywhere you want ♥
##############################################################################

def layout(app: dash.Dash) -> dbc.Container:
    """Return the Google Ads layout container."""
    return get_google_ads_tab(app)


def register_callbacks(app: dash.Dash) -> None:
    """Callbacks are registered at import time via dash.callback decorators."""
    pass

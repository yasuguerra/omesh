from __future__ import annotations

"""Dash entry point for Omesh Super-Dashboard."""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input

from .ui.navigation import navbar, root_tabs
from .pages import sales, google_analytics, google_ads, social

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Omesh Super-Dashboard"

app.layout = dbc.Container([
    navbar(),
    root_tabs(),
    html.Div(id="tab-content"),
], fluid=True)


@app.callback(Output("tab-content", "children"), Input("root-tabs", "value"))
def render_tab(tab: str):
    if tab == "sales":
        return sales.layout()
    if tab == "ga":
        return google_analytics.layout()
    if tab == "ads":
        return google_ads.layout(app)
    if tab == "social":
        return social.layout()
    return html.P("Unknown tab")


# Register callbacks for each page
sales.register_callbacks(app)
google_analytics.register_callbacks(app)
google_ads.register_callbacks(app)
social.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8052)


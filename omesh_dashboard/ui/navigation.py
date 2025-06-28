from __future__ import annotations

"""Navigation helpers for the Super-Dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def navbar() -> dbc.NavbarSimple:
    return dbc.NavbarSimple(
        brand="Omesh Super-Dashboard",
        color="dark",
        dark=True,
        className="mb-4",
    )


def root_tabs() -> dcc.Tabs:
    return dcc.Tabs(id="root-tabs", value="sales", children=[
        dcc.Tab(label="Sales", value="sales"),
        dcc.Tab(label="Google Analytics", value="ga"),
        dcc.Tab(label="Google Ads", value="ads"),
        dcc.Tab(label="Social", value="social"),
    ])


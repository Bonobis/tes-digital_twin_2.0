"""Dash application factory placeholder."""
from __future__ import annotations

import dash
from dash import dcc, html


def create_dash_app(server=None) -> dash.Dash:
    app = dash.Dash(__name__, server=server, title="Thermal Twin")
    app.layout = html.Div(
        [
            html.H1("Thermal Twin UI"),
            dcc.Markdown("Live simulation monitoring coming soon."),
        ]
    )
    return app

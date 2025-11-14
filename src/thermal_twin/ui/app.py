"""Dash application factory placeholder."""
from __future__ import annotations

from typing import Any

import dash
from dash import dcc, html


def create_dash_app(server: Any | None = None) -> dash.Dash:
    if server is None:
        app = dash.Dash(__name__, title="Thermal Twin")
    else:
        app = dash.Dash(__name__, server=server, title="Thermal Twin")
    app.layout = html.Div(
        [
            html.H1("Thermal Twin UI"),
            dcc.Markdown("Live simulation monitoring coming soon."),
        ]
    )
    return app

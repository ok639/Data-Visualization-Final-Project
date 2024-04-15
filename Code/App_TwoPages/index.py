

import dash 
from dash import Dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import dcc, html, dash_table, callback, callback_context, clientside_callback
from dash import Input, Output, State, MATCH, ALL


from app import app
from app import initializer, dict_classes, list_tickers, main_df
from Pages import Home, Analytics, Portfolio

from utils import * 
from utils_analytics import *






#####APP 


from dash import Dash
import dash_bootstrap_components as dbc

#Creating navigation bar : 
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
    ],
    brand="MOMENTUM, VOLATILITY & CORRELATIONS IN FINANCIAL MARKETS - AN EXPLORATIVE (DEMO) DASHBOARD",
    brand_href="/",
    color="primary",
    dark=True,
)


# Define the app layout with different pages
app.layout = html.Div([
    dcc.Store(id='session-data'),
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'), 
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/analytics':
        return layout_analytics
    elif pathname == '/portfolio':
        return layout_portfolio
    elif pathname == '/':
        return layout_home
    else:
        return html.Div([
            html.H1('404 Error'),
            html.P('Page not found: the pathname was {}'.format(pathname))
        ], style={'textAlign': 'center'})

        


#RUN IT ALL 

import webbrowser

# Run the app
port = 3003

# Open a web browser tab using the specified port
def open_browser():
      webbrowser.open_new_tab(f'http://127.0.0.1:{port}')

if __name__ == '__main__':
    # Use the threading module to open a web browser tab
    # This prevents blocking the execution of the app
    from threading import Timer
    Timer(1, open_browser).start()  # Wait 1 second before opening the tab
    
    app.run_server(debug=True, port=port)
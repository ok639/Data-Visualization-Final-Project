import time
import re 

from functools import reduce
from datetime import datetime as dt

import yfinance as yf 

import numpy as np 
import pandas as pd 
import networkx as nx
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler

import dash 
from dash import Dash 
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import dcc, html, dash_table, callback, callback_context, clientside_callback
from dash import Input, Output, State, MATCH, ALL


from app import app 
from app import initializer, dict_classes, list_tickers, main_df

from utils import *
from utils_analytics import *




layout_portfolio = html.Div([
    dcc.Location(id='url'),
    dcc.Store(id='ticker-list'),
    dcc.Store(id='store-selected-date'),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerSingle(
                id='portfolio-date-picker',
                display_format='YYYY-MM-DD'
            ),
            html.Button("Reset All", id="reset-button", className="btn btn-warning", style={'margin-top': '10px'}),  # Reset button next to DatePicker
        ], width=4),
        dbc.Col([
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[],  # dynamically generated options from list_ticker (global var)
                multi=False
            ),
            html.Button("Confirm Weight", id="confirm-weight-btn"),
            dcc.Slider(
                id='weight-slider',
                min=0,
                max=100,
                step=1,
                value=0,
                marks={i: str(i) for i in range(0, 101, 10)}
            ),
            html.Div(id='remaining-weight', children='Remaining Weight: 100', style={'margin-top': '20px'})
        ], width=8)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='portfolio-performance-graph'),  # Line chart for portfolio performance
        ], width=6),
        dbc.Col([
            dcc.Graph(id='stock-performance-viz'),  # Bar chart for stock performance
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='weights-table',
                columns=[
                    {'name': 'Ticker', 'id': 'ticker'},
                    {'name': 'Weight', 'id': 'weight'}
                ],
                data=[]
            )
        ])
    ]),
    html.Button("Launch Portfolio Analysis", id="launch-analysis-btn"),
])








counter = 0 
@app.callback(
    Output('ticker-dropdown', 'options'),
    Input('url', 'pathname'),
)
def update_dropdown_options(pathname):
    global list_tickers, counter
    
    if pathname.endswith('/portfolio') and counter == 0:
        print(list_tickers)
        data = [{'label': ticker, 'value': ticker} for ticker in list_tickers]
        seen = set()
        options = []
        for item in data:
        # Define a tuple of characteristics to check for uniqueness
            identifier = (item['label'], item['value'])
            if identifier not in seen:
                options.append(item)
                seen.add(identifier)
        
        print(f"Dropdown options updated: {options}")
        counter += 1
        return options
    return []


@app.callback(
    [
        Output('remaining-weight', 'children'),
        Output('weight-slider', 'max'),
        Output('weight-slider', 'value'),
        Output('weights-table', 'data'),
        Output('weights-table', 'columns')
    ],
    [   Input('confirm-weight-btn', 'n_clicks'),
        Input('reset-button', 'n_clicks')],
    [
        State('weight-slider', 'value'),
        State('ticker-dropdown', 'value'),
        State('weights-table', 'data'),
        State('weight-slider', 'max')]
)
def confirm_weight(n_clicks, reset, slider_value, ticker, data, current_max):
#     print(list_tickers)
    print(f"Confirm weight called with: {n_clicks}, {slider_value}, {ticker}, {current_max}")
    if n_clicks is None:
        raise PreventUpdate

    # Avoid duplicate entries if button is clicked more than once quickly
    if data and data[-1]['ticker'] == ticker and data[-1]['weight'] == slider_value:
        return dash.no_update

    data.append({'ticker': ticker, 'weight': slider_value})
    new_max = current_max - slider_value
    
    if reset: 
        new_max = 100 
        return f'Remaining Weight: {new_max}', new_max, 0, data, []
    
    return f'Remaining Weight: {new_max}', new_max, 0, data, [{'name': 'Ticker', 'id': 'ticker'}, {'name': 'Weight', 'id': 'weight'}]


@app.callback(
    Output('portfolio-performance-graph', 'figure'),
    Output('stock-performance-viz', 'figure'),
    Input('launch-analysis-btn', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('weights-table', 'data')
)
def update_graphs(n_clicks, reset, data):
    print(f"Update graphs called with: {n_clicks}, data: {data}")
    if n_clicks is None or not data:
        raise PreventUpdate

    if reset == 0: 
        line_chart = go.Figure()
        line_chart.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 3, 4], mode='lines', name='Portfolio Value'))
        line_chart.update_layout(title='Portfolio Performance Over Time')

        bar_chart = go.Figure()
        bar_chart.add_trace(go.Bar(x=[item['ticker'] for item in data], y=[item['weight'] for item in data], name='Stock Weight'))
        bar_chart.update_layout(title='Stock Performance Visualization')

        return line_chart, bar_chart
    
    else: 
        reset_graph1 = {'data': []}
        reset_graph2 = {'data': []}
        
        return line_chart, bar_chart


@callback(
        Output('portfolio-date-picker', 'date'),
        Input('reset-button', 'n_clicks'),
        Input('store-selected-date', 'data')
)
def reset_all(n_clicks, session_data):
    if n_clicks is None:
        return no_update

    reset_date = session_data.get('date')

    return reset_date

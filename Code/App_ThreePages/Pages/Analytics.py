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
from dash import dcc, html


from app import app 
from app import initializer, dict_classes, list_tickers, main_df

from utils import *
from utils_analytics import *





layout_analytics = html.Div([
    html.H1("Directional Momentum Visualization", style={'textAlign': 'center'}),
    html.H2("Network Graph Visualization", style={'textAlign': 'center'}),
    dcc.Store(id='store-corr-threshold', storage_type='session'),  # Store for correlation threshold
    dcc.Store(id='store-span', storage_type='session'),  # Store for span
    dcc.Store(id='store-selected-date', storage_type='session'),  # Store for selected date
    dcc.Store(id='net-graph', storage_type='session'),
    dcc.Store(id='heatmap-relative', storage_type='session'),
    dcc.Store(id='heatmap-directional', storage_type='session'),

    html.Div([
        html.Div([
            html.Label("Correlation Threshold", style={'textAlign': 'center'}),
            dcc.Slider(
                id='corr-threshold-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.5,  # Default value
                marks={str(i/10): str(i / 10) for i in range(0, 11)}, 
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

        html.Div([
            html.Label("Span", style={'textAlign': 'center'}),
            dcc.Input(
                id='span-input',
                type='number',
                value=5,  
                min=1,  
                max=100,  
                step=1  
            )
        ], style={'width': '15%', 'display': 'inline-block', 'padding': '20px'}),

        html.Div([
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed='start',
                max_date_allowed='end',
                initial_visible_month='start',
                date=str('start')  # Set initial date
            )
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'right', 'float': 'right', 'padding': '20px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    

    dcc.Graph(id='network-graph'),
    html.Div([
        html.Div([
            html.H2("Heatmap, Relative", style={'textAlign': 'center'}),
            dcc.Graph(id='heatmap-relative'),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H2("Heatmap, Directional", style={'textAlign': 'center'}),
            dcc.Graph(id='heatmap-directional'),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),  # This container will hold both heatmaps side by side
    
    html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold'})  # Error message div
], style={'padding': '20px'})










@app.callback(
    Output('span-input', 'value'),
    [Input('date-picker', 'date'),
     Input('span-input', 'value')]
)
def debugging(ref_date, span):
    
    #First : Check if Span input is indeed a positive int
    assert isinstance(span, int)
    assert span >= 0, "span must be a positive integer"
    
    #Second : Debugging the Ref_Date 
    if ref_date not in main_df.index:
        print('Ref_date not in df.index')
        ref_date = main_df.index[0]
    else: 
        ref_date = ref_date 
    
    #Third : Debugging the Span  
    #Retrieve position of the current date
    position = main_df.index.get_loc(ref_date) + 1
    if 2*span > position: #because we need 2 periods of length span 
        print('Span too high compared to index position. Rescaling')
        span = position // 2

    print(f'Select position : {main_df.index[span]} minimum for a span window of {span}.')

    return span


@app.callback(
    [Output('network-graph', 'figure'),
     Output('heatmap-relative', 'figure'),
     Output('heatmap-directional', 'figure'),
     Output('store-corr-threshold', 'data'),
     Output('store-span', 'data'),
     Output('store-selected-date', 'data'),
     Output('error-message', 'children')],
    [Input('date-picker', 'date'),
     Input('corr-threshold-slider', 'value'), 
     Input('span-input', 'value')]
)
def update_graph(selected_date, corr_threshold, span):
    global main_df
    
    #Only selecting the columns which are named as "Ticker : Log-Returns" from main_df
    suffix = 'Log-Returns'
    subset_cols = [col for col in list(main_df.columns) if col.endswith(suffix)]
    subset_df = main_df[subset_cols]
    subset_df.columns = list_tickers
    
    
    span = debugging(selected_date, span)
    
    if selected_date is not None and corr_threshold is not None:
        # Check if selected_date is valid using rolling_corr or equivalent function
        corr_result = rolling_corr(subset_df, selected_date, span)  
        if isinstance(corr_result, str) and corr_result == "Date not found":
            return dash.no_update, dash.no_update, dash.no_update, corr_threshold, span, {'date' : selected_date}, "Error: Selected date not found in the dataset."
        else:
            try: 
                network_fig = graph_net(subset_df, selected_date, corr_threshold=corr_threshold, span=span)
                heatmap_relative_graph, heatmap_directional_graph = rolling_corr_difference(subset_df, selected_date, span=span)
                return network_fig, heatmap_relative_graph, heatmap_directional_graph, corr_threshold, span, selected_date, ""
            
            except Exception as e:
                return dash.no_update, dash.no_update, dash.no_update, corr_threshold, span, {'date' : selected_date}, f"Error: {str(e)}"
    else:
        return dash.no_update, dash.no_update, dash.no_update, corr_threshold, span, {'date' : selected_date}, "" 

    
@app.callback(
    [Output('date-picker', 'min_date_allowed'),
     Output('date-picker', 'max_date_allowed'),
     Output('date-picker', 'initial_visible_month'),
     Output('date-picker', 'date')],
    [Input('session-data', 'data')],
)
def update_date_picker(session_data):
    if session_data:
        min_date = session_data.get('start')
        max_date = session_data.get('end')
        return min_date, max_date, min_date, min_date

    default_date = datetime.datetime.today().date()
    return default_date, default_date, default_date, default_date


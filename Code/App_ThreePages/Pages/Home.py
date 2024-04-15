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







from dash import dcc, html
import dash_bootstrap_components as dbc

layout_home = html.Div([
    dcc.Store(id='storage'),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.H3("Enter Stock Ticker:", style={'textAlign': 'center'}),
                dbc.Input(id='ticker-input', placeholder='Enter ticker, e.g., AAPL', type='text', value='AAPL', style={'margin': '30px 0'})
            ], style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'}), width=12)
        ], justify='center', className="mb-4", style={'paddingTop': '10px'}),

        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='asset-class-dropdown',
                    options=[
                        {'label': 'Equity', 'value': 'Equity'},
                        {'label': 'Forex', 'value': 'Forex'},
                        {'label': 'Fixed Income', 'value': 'Fixed Income'},
                        {'label': 'Commodities', 'value': 'Commodities'}
                    ],
                    placeholder="Select asset class", value="Equity"), width=4),
            dbc.Col(
                dbc.Input(id='start-date-input', placeholder='Start Date (YYYY-MM-DD)', type='text', value='2020-12-31'), width=4),
            dbc.Col(
                dbc.Input(id='end-date-input', placeholder='End Date (YYYY-MM-DD)', type='text', value='2023-12-31'), width=4)
        ], justify='center', className="mb-4"),  # Added comma here

        dbc.Row([
            dbc.Col([
                dbc.Checkbox(id='scale-checkbox', className='form-check-input'),
                html.Label('Scaled', htmlFor='scale-checkbox', className='form-check-label', style={'margin-left': '10px'})
            ], width=3, align='start'),
            dbc.Col(dbc.Button('Download Data', id='submit-button', color='danger', n_clicks=0, className='btn-lg'), width=3, align='center'),
            dbc.Col(dbc.Button('Reset Data', id='reset-button', color='primary', n_clicks=0, className='btn-lg'), width=3, align='end')
        ], justify='center', className="mb-3"),  # Added comma here

        dbc.Row([
            dbc.Col(html.Div(id='output-container', style={'color': 'white'}), width=12)
        ], justify='center', className="mb-3")
    ], style={'height': '100vh', 'backgroundColor': '#000000', 'color': 'white'})
], style={'backgroundColor': '#000000'})


# clientside_callback(
#     """
#     function(n_clicks, start_date, end_date) {
#         if(n_clicks > 0) {
#             sessionStorage.setItem('start_date', start_date);
#             sessionStorage.setItem('end_date', end_date);
#         }
#     }
#     """,
#     Output('url', 'pathname'),  # Navigate or refresh the page as needed
#     [Input('save-dates-btn', 'n_clicks')],
#     [State('start-date-input', 'date'), State('end-date-input', 'date')]
# )







# @app.callback(
#     Output('storage-data', 'data'),
#     [Input('start-date-input', 'value'),
#      Input('end-date-input', 'value'),
#      Input('scale-checkbox', 'value')],
#     prevent_initial_call=True
# )
# def store_date_and_scale(start_date, end_date, scale):
#     return {'start': start_date, 'end': end_date, 'scale': scale}

# @app.callback(
#     [Output('start-date-input', 'value'),
#      Output('end-date-input', 'value'),
#      Output('scale-checkbox', 'value')],
#     [Input('storage-data', 'data')],
#     prevent_initial_call=True
# )
# def load_date_and_scale(storage_data):
#     if storage_data is None:
#         raise exceptions.PreventUpdate
#     return storage_data.get('start'), storage_data.get('end'), storage_data.get('scale')


@app.callback(
    [Output('output-container', 'children'),
     Output('session-data', 'data')],
    [Input('submit-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('scale-checkbox', 'value'),
     Input('session-data', 'data')],
    [State('ticker-input', 'value'),
     State('asset-class-dropdown', 'value'),
#      State('sector-dropdown', 'value'),
     State('start-date-input', 'value'),
     State('end-date-input', 'value')]
)
def update_or_reload_data(submit_n_clicks, reset_n_clicks, scale, session_data,
                          ticker, asset_class, start_date, end_date):
    global main_df
    global initializer
    global list_tickers, dict_classes, dict_sectors, dict_commo
    key = 0
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    json_dump = None
    
    start_time = dt.strptime(start_date, "%Y-%m-%d").date()
    end_time = dt.strptime(end_date, "%Y-%m-%d").date()

    if initializer:
        initializer = False
        return html.Div("Empty Graph Data"), {'Price Plot' : None, 'start' : None, 'end' : None}
    
    if triggered_id == 'reset-button' and reset_n_clicks > 0:
        main_df = pd.DataFrame()
        reset_global_vars()
        return html.Div("Data has been reset"), {'Price Plot' : None, 'start' : None, 'end' : None}

    
    if triggered_id == 'submit-button' or triggered_id == 'scale-checkbox':
        try:
            df = data_loader(ticker, start=start_date, end=end_date)
            df.dropna(inplace=True)
            if df.empty:
                return html.Div("No data available for the selected ticker and date range."), {'Price Plot' : None, 'start' : None, 'end' : None}

            if main_df.empty:
                main_df = df
            else:
                prefix = ticker
                matching_columns = [col for col in main_df.columns if col.startswith(prefix)]
                main_df.drop(columns=matching_columns, inplace=True)
                if len(df) <= len(main_df):
                    #Update the old df by merging with new df - INNER JOIN 
                    main_df = pd.merge(main_df, df, left_index=True, right_index=True, how='inner')
                else: 
                    data_loader_format_all(ticker, start_date, end_date)

        except Exception as e:
            return html.Div(f"Failed to load data for {ticker}: {str(e)}"), {'Price Plot' : None, 'start' : None, 'end' : None}  
    
    elif session_data:
        if 'Price Plot' in session_data and session_data['Price Plot']:
            json_str = session_data['Price Plot']
            try:
                json_dump = pd.read_json(json_str, orient='split')
                key = 1 
            except ValueError as e:
                print("Error loading JSON data:", e)
                return html.Div("Failed to load data."), {'Price Plot': None, 'start': None, 'end': None}

    #Select subset of main_df 
    sub_df = prepare_and_scale_df(main_df, json_dump, key, scale)
    
    fig = plot_fig(sub_df)

    #UPDATE TO GLOBAL VARIABLES
    list_tickers.append(ticker)
    list_tickers = list(set(list_tickers))

    dict_classes[asset_class].append(ticker)
    dict_classes[asset_class] = list(set(dict_classes[asset_class]))

#     if asset_class == 'Commodities':
#         dict_commo[asset_sector].append(ticker)
#         dict_commo[asset_sector] = list(set(dict_commo[asset_sector]))
#     if asset_class == 'Equity':
#         dict_sectors[asset_sector].append(ticker)
#         dict_sectors[asset_sector] = list(set(dict_sectors[asset_sector]))

    return dcc.Graph(figure=fig), {'Price Plot' : sub_df.to_json(date_format='iso', orient='split'), 
                                   'start' : start_time, 
                                   'end' : end_time}



# @app.callback(
#     Output('sector-dropdown', 'options'),
#     [Input('asset-class-dropdown', 'value')]
# )
# def set_sectors_options(selected_asset_class):
#     if selected_asset_class == 'Equity':
#         return [{'label': 'Technology', 'value': 'Technology'},
#                 {'label': 'Consumer', 'value': 'Consumer'},
#                 {'label': 'Utilities', 'value': 'Uitilies'},
#                 {'label': 'Industrial', 'value': 'Consumer'},
#                 {'label': 'Financials', 'value': 'Financials'},
#                 {'label': 'Other', 'value': 'Other'},]
#     elif selected_asset_class == 'Commodities':
#         return [{'label': 'Agriculture', 'value': 'Agriculture'},
#                 {'label': 'Precious Metals', 'value': 'Precious Metals'},
#                 {'label': 'Industrials', 'value': 'Industrials'},
#                 {'label': 'Energy', 'value': 'Energy'}]
#     else:
#         return [{'label' : 'N/A', 'value' : 'N/A'}]


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





from dash import dcc, html
import dash_bootstrap_components as dbc

layout_home = html.Div([
#     dcc.Store(id='session-data', data={})
    html.H2("Home Page", style={'textAlign': 'left', 'color':'white'}),
    html.H2("Construct your custom Dataframe:", style={'textAlign': 'center', 'color':'white'}),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.H3("Enter Stock Ticker:", style={'textAlign': 'center', 'color' : 'blue'}),
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
                        {'label': 'Commodities', 'value': 'Commodities'}, 
                        {'label' : 'Index' , 'value' : 'Index'},
                    ], placeholder="Select asset class", value="Equity", 
                    style={'backgroundColor': '#f0f0f0', 'color': 'black'}), width=4),
            dbc.Col(
                dbc.Input(id='start-date-input', placeholder='Start Date (YYYY-MM-DD)', type='text', value='2000-01-01'), width=4),
            dbc.Col(
                dbc.Input(id='end-date-input', placeholder='End Date (YYYY-MM-DD)', type='text', value=dt.today().strftime('%Y-%m-%d')), width=4)
        ], justify='center', className="mb-4"), 

        dbc.Row([
            dbc.Col([
                dbc.Checkbox(id='scale-checkbox', className='form-check-input'),
                html.Label('Scaled', htmlFor='scale-checkbox', className='form-check-label', style={'margin-left': '10px'})
            ], width=3, align='start'),
            dbc.Col(dbc.Button('Download Data', id='submit-button', color='danger', n_clicks=0, className='btn-lg'), width=3, align='center'),
            dbc.Col(dbc.Button('Reset Data', id='reset-button', color='primary', n_clicks=0, className='btn-lg'), width=3, align='end')
        ], justify='center', className="mb-3"),

        dbc.Row([
            dbc.Col(html.Div(id='output-container', style={'color': 'white'}), width=12)
        ], justify='center', className="mb-3")
    ], style={'height': '100vh', 'backgroundColor': '#000000', 'color': 'white'})
], style={'backgroundColor': '#000000'})


@app.callback(
    [Output('output-container', 'children'),
     Output('session-data', 'data')],
    [Input('submit-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('scale-checkbox', 'value'),
     Input('session-data', 'data'),
     Input('url', 'pathname')],  # Listen to URL changes
    [State('ticker-input', 'value'),
     State('asset-class-dropdown', 'value'),
#      State('sector-dropdown', 'value'),
     State('start-date-input', 'value'),
     State('end-date-input', 'value')]
)
def update_or_reload_data(submit_n_clicks, reset_n_clicks, scale, jsonified_data, pathname,
                          ticker, asset_class, start_date, end_date):
    global main_df
    global initializer
    global list_tickers, dict_classes, dict_sectors, dict_commo
    key = 0
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

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
#                 main_df = pd.merge(main_df, df, left_index=True, right_index=True, how='inner')
                if len(df) <= len(main_df):
                    #Update the old df by merging with new df - INNER JOIN 
                    main_df = pd.merge(main_df, df, left_index=True, right_index=True, how='inner')
                else: 
                    data_loader_format_all(ticker, start_date, end_date)
        except Exception as e:
            return html.Div(f"Failed to load data for {ticker}: {str(e)}"), {'Price Plot' : None, 'start' : None, 'end' : None}  
    
    elif jsonified_data:
        if 'Price Plot' in jsonified_data and jsonified_data['Price Plot']:
            json_str = jsonified_data['Price Plot']
            try:
                json_dump = pd.read_json(json_str, orient='split')
                key = 1 
            except ValueError as e:
                print("Error loading JSON data:", e)
                return html.Div("Failed to load data."), {'Price Plot': None, 'start': None, 'end': None}

    # Subset DataFrame focusing on 'Adj Close'
    if key == 0:
        col_list = list(main_df.columns)
        adj_close_list = [item for item in col_list if "Adj Close" in item]
        sub_df = main_df[adj_close_list]
        sub_df.columns = [re.sub(r'^\s+|\s+$', '', col.split(":")[0]) for col in adj_close_list]
    else:
        col_list = list(json_dump.columns)
        adj_close_list = [item for item in col_list if "Adj Close" in item]
        sub_df = json_dump[adj_close_list]
        sub_df.columns = [re.sub(r'^\s+|\s+$', '', col.split(":")[0]) for col in adj_close_list]

    if scale and not sub_df.empty:
        sub_df = scaled_df(sub_df)

    # Plotting the graph
    fig = go.Figure()
    for ticker in sub_df.columns:
        fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df[ticker], mode='lines', name=ticker))

    fig.update_layout(
        title='Adjusted Close Prices Over Time for selected tickers',
        xaxis_title='Date',
        yaxis_title='Adjusted Close Price',
        legend_title="Ticker"
    )

    start_time = dt.strptime(start_date, "%Y-%m-%d").date()
    end_time = dt.strptime(end_date, "%Y-%m-%d").date()

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
    
#     return "Enter values and click 'Download Data' or 'Reset Data'.", {'Price Plot' : None, 'start' : None, 'end' : None} 





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
    

    
# @app.callback(
#     Output('sector-dropdown', 'value'),
#     [Input('asset-class-dropdown', 'value')]
# )
# def set_default_sector(selected_asset_class):
#     if selected_asset_class == 'Equity':
#         return 'Technology'  
#     elif selected_asset_class == 'Commodities':
#         return 'Agriculture'  
#     return None  

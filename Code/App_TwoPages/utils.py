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










def scaled_df(df):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    return scaled_df


def get_logreturn(df_price_col):
    return np.log(df_price_col / df_price_col.shift(1)) * 100 


def minmax_scaler(df, spread = 1):
    min_df = df.min()
    max_df = df.max() 
    
    df = df.apply(lambda x : ((x-min_df) / (max_df - min_df)) / spread)
    
    return df 


def data_loader(ticker_name, start, end, interval='1d'):
    ticker = ticker_name 
    try : 
        df = yf.download(ticker, start=start, end=end)
        
    
        if len(df) != 0: 
            date_column = df.index
            df = df.sort_index()

            #Daily price normalized to the total volume of the dataset
            df['price_volume'] = df['Adj Close'] * df['Volume']
            df['Norm_PV'] = (df['price_volume'] / df['Volume'].sum()) #Norm_PV = Normalized Price Volume

            #Renaming (for merging purposes):
            title_1 = ticker + ': Adj Close'
            title_2 = ticker + ': Norm_PV'
            title_3 = ticker + ': Log-Returns'

            #Log-returns
            df[title_3] = get_logreturn(df['Adj Close']) #Results are in %
            df.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'price_volume'], axis = 1, inplace = True)

            df.rename(columns={'Close': title_1,
                                    'Norm_PV': title_2,
                                   }, inplace=True)
            pd.to_datetime(df.index, format='%Y-%m-%d')

        return df 
    
    except Exception as e: 
        return e



def data_loader_format_all(ticker, start, end, interval='1d'):
    global main_df, list_tickers
    
    main_df = pd.DataFrame()

    temp = list_tickers 
    temp.append(ticker)
    temp = set(temp)
    
    for i, tick in enumerate(temp):             
        if i == 0: 
            main_df = data_loader(tick, start, end)
            continue
        df = data_loader(tick, start, end)
        main_df = pd.merge(main_df, df, left_index=True, right_index=True, how='inner')

    return "All ticker frames updated." 


def curve_plotter(df, mode='price', scaled=False):
    global list_tickers 
    if mode == 'return':
        scaled=False
        suffix='Returns'
    else: 
        suffix='Adj Close'
        
    matching_columns = [col for col in df.columns if col.endswith(suffix)]
    
    temp_df = pd.DataFrame()

    df.drop(matching_columns, axis=1, inplace=True)  
    
    
def prepare_and_scale_df(df1, df2, key, scale):
    if key == 0:
        col_list = list(df1.columns)
        adj_close_list = [item for item in col_list if "Adj Close" in item]
        sub_df = df1[adj_close_list]
        sub_df.columns = [re.sub(r'^\s+|\s+$', '', col.split(":")[0]) for col in adj_close_list]
    else:
        col_list = list(df2.columns)
        adj_close_list = [item for item in col_list if "Adj Close" in item]
        sub_df = df2[adj_close_list]
        sub_df.columns = [re.sub(r'^\s+|\s+$', '', col.split(":")[0]) for col in adj_close_list]

    if scale and not sub_df.empty:
        sub_df = scaled_df(sub_df)
    
    return sub_df


def plot_fig(df):
    fig = go.Figure()
    for ticker in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=ticker))

    fig.update_layout(
        title='Adjusted Close Prices Over Time for selected tickers',
        xaxis_title='Date',
        yaxis_title='Adjusted Close Price',
        legend_title="Ticker"
    )
    
    return fig

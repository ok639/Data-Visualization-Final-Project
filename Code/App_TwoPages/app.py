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



### GLOBAL VARS

main_df = pd.DataFrame()
available_dates = main_df.index.tolist()

initializer = True 

dict_classes = {
    'Equity' : [],
    'Commodities' : [],
    'Fixed Income' : [],
    'Forex' : [],
}

dict_sectors = {
    'Financials' : [],
    'Technology' : [],
    'Industrial' : [],
    'Consumer' : [],
    'Utilities' : [],
    'Other' : []
}

dict_commo = {
    'Agriculture' : [],
    'Precious Metals' : [],
    'Energy' : [],
    'Industrials' : []
}

list_tickers = ['AAPL']




#####APP 

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__,
           external_stylesheets=external_stylesheets,
           suppress_callback_exceptions=True,  
           prevent_initial_callbacks=True) 




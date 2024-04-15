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






def relative_change(corr1, corr2):
    range_corr = 2
    rel_range_change = ((corr2 - corr1) / range_corr) * 100
    
    return rel_range_change        



def rolling_corr(df, ref_date, span):
    try:
        position = df.index.get_loc(ref_date) + 1
        start_position = position - span
        filtered_df = df.iloc[start_position:position]
        corr_matrix = filtered_df.corr()

        return corr_matrix.round(2)

    except KeyError as err: 
        print(f'Error due to wrong date/span input: {err}, Date : {ref_date}, Span : {span}.')
        print(f'Recall date range of input dataframe: {df.index[0], df.index[-1]}')

    

def matrix_difference(matrix1, matrix2):
    # Check if the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")
    
    # Initialize an empty matrix to store the differences
    rows = matrix1.shape[0]
    columns = matrix1.shape[1]
    result_matrix = pd.DataFrame(np.zeros((rows, columns)))
    result_matrix.columns = list_tickers
    
    # Iterate through the rows and columns of the matrices
    for i in range(rows):
        for j in range(columns):
            perct_change = relative_change(matrix1.iloc[i, j], matrix2.iloc[i, j])
            result_matrix.iloc[i, j] = perct_change       
    return result_matrix



def matrix_difference_qual(matrix1, matrix2, heatmap=True):
    category_map = None

    # Check if the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")
    
    # Initialize an empty matrix to store the differences
    rows = matrix1.shape[0]
    columns = matrix1.shape[1]
    result_matrix = pd.DataFrame(np.zeros((rows, columns)))
    result_matrix.columns = list_tickers
    
    # Iterate through the rows and columns of the matrices
    for i in range(rows):
        for j in range(columns):
            # Perform differentiation based on the values of the elements
            
            if matrix1.iloc[i, j] == 0 or matrix2.iloc[i, j] == 0:
                if matrix1.iloc[i, j] < 0:
                    matrix2.iloc[i,j] = -0.00001
                elif matrix1.iloc[i,j] > 0: 
                    matrix2.iloc[i,j] = 0.00001
                elif matrix2.iloc[i,j] < 0: 
                    matrix1.iloc[i,j] = 0.00001
                elif matrix2.iloc[i,j] > 0: 
                    matrix1.iloc[i,j] = -0.00001
                    
            if matrix1.iloc[i, j] < 0 and matrix2.iloc[i, j] < 0:
                if matrix2.iloc[i, j] > matrix1.iloc[i, j]:
                    result_matrix.iloc[i, j] = 'Neg Stronger'
                else:
                    result_matrix.iloc[i, j] = 'Neg Weaker'
            elif matrix1.iloc[i, j] > 0 and matrix2.iloc[i, j] > 0:
                if matrix2.iloc[i, j] > matrix1.iloc[i, j]:
                    result_matrix.iloc[i, j] = 'Pos Stronger'
                else:
                    result_matrix.iloc[i, j] = 'Pos Weaker'
            elif matrix1.iloc[i, j] > 0 and matrix2.iloc[i, j] < 0:
                result_matrix.iloc[i, j] = 'Neg Stronger'
            elif matrix1.iloc[i, j] < 0 and matrix2.iloc[i, j] > 0:
                result_matrix.iloc[i, j] = 'Pos Stronger'
            elif 0.95 <= matrix1.iloc[i, j] / matrix2.iloc[i, j] <= 1.05:
                result_matrix.iloc[i, j] = 'UNCH'
            else: 
                print(matrix1.iloc[i, j], matrix2.iloc[i,j])
                print(list_tickers[i], list_tickers[j])
                
    if heatmap: 
        category_map = {'Neg Stronger': -10, 'Neg Weaker': -5, 
                        'Pos Stronger': 10, 'Pos Weaker': 5, 
                        'UNCH': 0}
        df_numeric = result_matrix.applymap(lambda x: category_map[x])
        df_numeric.index = list_tickers
        
        return df_numeric

    else: 
        print("Can't return a heatmap - Categ Variables of String Type")
        return result_matrix

    

def rolling_corr_difference(df, ref_date, span):
    
    # Calculate correlation matrix for the current span
    corr_matrix_current = rolling_corr(df, ref_date, span)
    
    # Get the previous corr matrix's ref_date
    index_position = df.index.get_loc(ref_date)
    
    # Previous corr matrix's ref index position is max(0, index_position - span)
    temp_index_position = index_position - span 
    if temp_index_position < 0: 
        print(f'Period for Previous Corr Matrix calculation out of bound. Setting reference date to {df.index[0]}')
        span = index_position

    new_index_position = df.index.get_loc(df.index[temp_index_position])
    new_ref_date = df.index[new_index_position]

    # Calculate correlation matrix for the previous span
    corr_matrix_prev = rolling_corr(df, new_ref_date, span)
    
    # Calculate the difference between correlation matrices
    corr_diff = matrix_difference(corr_matrix_prev, corr_matrix_current)
    corr_diff_qual = matrix_difference_qual(corr_matrix_prev, corr_matrix_current)
    
    column_names = df.columns.tolist()
    
    #MASKING FOR HALF HEAT MAPs
    mask = np.triu(np.ones_like(corr_diff, dtype=bool))
    corr_diff_masked = np.where(mask, None, corr_diff)  # Replace upper triangular part with None
    
    mask_qual = np.triu(np.ones_like(corr_diff_qual, dtype=bool))
    corr_diff_qual_masked = np.where(mask_qual, None, corr_diff_qual)  # Replace upper triangular part with None

    
    # Plotting heatmap for relative range percentage change
    fig_relative = go.Figure(data=go.Heatmap(z=corr_diff_masked, colorscale='RdYlGn',
                                             x=column_names, y=column_names))
    fig_relative.update_layout(title=f'Relative Range Percentage Change of Rolling Correlations between Assets Log Returns, {span} freq periods.',
                               xaxis_title='Assets', yaxis_title='Assets',                           
                               plot_bgcolor='white',  paper_bgcolor='white')

    # Assuming `corr_diff_qual` is plotted here instead
    fig_directional = go.Figure(data=go.Heatmap(z=corr_diff_qual_masked, colorscale='bluered',
                                                x=column_names, y=column_names))
    fig_directional.update_layout(title=f'Directional Difference between Rolling Correlation Matrices of Assets Log Returns, {span} freq periods.',
                                  xaxis_title='Assets', yaxis_title='Assets',
                                  plot_bgcolor='white',  paper_bgcolor='white')
    # Customizing the colorbar tick labels
    fig_directional.update_traces(colorbar_tickvals=[-10, -5, 0, 5, 10],
                  colorbar_ticktext=['Negative Stronger', 'Negative Weaker', 'UNCH', 'Positive Weaker', 'Positive Stronger'])
    
    fig_relative.update_layout(
    width=900,  # Adjust width
    height=900,  # Adjust height
    title=f'Relative Range Percentage Change of Rolling Correlations between Assets Log Returns, {span} freq periods.',
    xaxis_title='Assets',
    yaxis_title='Assets',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis={'autorange': True, 'tickangle': 45},  # Rotate x-axis labels to prevent overlap
    yaxis={'autorange': True}
    )

    fig_directional.update_layout(
        width=900,  # Adjust width
        height=900,  # Adjust height
        title=f'Directional Difference between Rolling Correlation Matrices of Assets Log Returns, {span} freq periods.',
        xaxis_title='Assets',
        yaxis_title='Assets',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis={'autorange': True, 'tickangle': 45},  # Rotate x-axis labels to prevent overlap
        yaxis={'autorange': True}
    )
    return fig_relative, fig_directional


# In[336]:


def graph_net(df, ref_date, corr_threshold, span):
    global dict_classes
    corr_matrix = rolling_corr(df, ref_date, span)
    
    # Initialize the graph
    G = nx.Graph()
    
    # Ticker categories and their colors
    ticker_categs = dict_classes
    
    colors = {
        'Equity': 'yellow',
        'Index': 'blue',
        'Fixed Income': 'red',
        'Commodities': 'orange',
        'Crypto': 'green'
    }
    
    # Node colors based on their category
    node_colors = {}
    for category, nodes in ticker_categs.items():
        for node in nodes:
            node_colors[node] = colors[category]
    
    # Determine which nodes should be included based on the correlation threshold
    nodes_to_include = set()
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and abs(corr_matrix.loc[i, j]) > corr_threshold:
                nodes_to_include.add(i)
                nodes_to_include.add(j)
    
    # Only add nodes that are part of an edge meeting the threshold
    for node in nodes_to_include:
        G.add_node(node, color=node_colors.get(node, 'grey'))
    
    # Add edges to the graph based on correlation
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j:  # Ensure we don't compare the same stock to itself
                corr = corr_matrix.loc[i, j]
                if abs(corr) > corr_threshold:  # Check if the correlation meets the threshold
                    # Add an edge with color based on the sign of the correlation
                    G.add_edge(i, j, weight=corr, color='green' if corr > 0 else 'red')

    # Assuming 'G' is your original graph with 'weight' attributes holding the correlations
    H = G.copy()

    # Update edge weights in H to be absolute values of the original weights
    for u, v, d in H.edges(data=True):
        d['weight'] = abs(d['weight'])


    # Assuming H is your graph for layout and G contains original correlation weights
    pos = nx.kamada_kawai_layout(H)                    
    

    # Initialize the figure once, before the loop
    fig = go.Figure()

    # For edges, create individual traces within the loop
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr_value = edge[2]['weight']

        # Determine the color based on the correlation value
        edge_color = 'green' if corr_value > 0 else 'red'

        # Create an individual trace for this edge
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            line=dict(width=0.5, color=edge_color),
            mode='lines',
            hoverinfo='none',
            showlegend=False# No hover info for the line itself
        )
        fig.add_trace(edge_trace)

        # Invisible marker at the midpoint for hover text
        midpoint_trace = go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            text=[f'{edge[0]}-{edge[1]}: {corr_value:.2f}'],
            mode='markers',
            hoverinfo='text',
            marker=dict(size=0.1, color='rgba(0,0,0,0)'),  # Make the marker virtually invisible
            showlegend=False
        )
        fig.add_trace(midpoint_trace)

        
    # Track which categories have been added to the legend
    added_categories = set()
    for node in G.nodes():
        x, y = pos[node]
        category = None
        for categ, members in ticker_categs.items():
            if node in members:
                category = categ
                break
        if category and category not in added_categories:
            # Add a representative node for this category to the legend
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(color=colors[category], size=10),
                name=category,  # This sets the legend entry,
                hoverinfo='none'
            ))
            added_categories.add(category)

    # Add node trace after all edge traces have been added
    node_x = []
    node_y = []
    node_text = []
    node_marker_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_marker_colors.append(G.nodes[node]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=node_text, mode='markers+text', hoverinfo='none',
        marker=dict(showscale=False, color=node_marker_colors, size=20, line_width=2),
        textposition="bottom center", showlegend=True
    )

    fig.add_trace(node_trace)

    # Set the layout for the figure
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend_title_text='Node Categories',
        legend=dict(x=1, y=0, xanchor='right', yanchor='bottom')
    )

    return fig

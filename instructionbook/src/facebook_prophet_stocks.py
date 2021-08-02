import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import altair as alt

import base64
from io import BytesIO

from src.end_code_block import end_code_block as __

def app():
    # example of forecasting
    st.header("Forecasting Stock Price Movements with Facebook Prophet")
    st.write('Step #1: import Python libraries')

    with st.echo():
        import urllib.request
        import pandas as pd
        import json
        import requests
        import datetime as dt
        import plotly.express as px
        from prophet import Prophet

    st.write('Step #2: load data with Pandas library')

    def load_dataset(data_link):
        dataset = pd.read_csv(data_link)
        return dataset
    
    # ticker = pd.read_csv('https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/stocks.csv')
    
    
    st.markdown('----')
    st.write('Create a variable named "ticker" to represent and store the stock prices as a dataframe. Note that your path to where the data is store may be different. The example show that the data is stored in GitHub.')
    
    __(display=False)
    # variable ticker stores google stock price data as a dataframe object
    ticker = pd.read_csv('https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/stocks.csv')
    __()    
    
    
    st.write('Show the first 5 rows using the "head" method available from Pandas:')
    __(display=False)
    ticker.head()
    __()
        
    st.dataframe(ticker.head())        
    
    st.write('Create a simple plot of Google Stock Price. Notes: fig is the name of the variable that holds the data to plot. Remember also that "px" is an alias for Plotly.')
    st.write('what we are asking Plotly to do is create a line chart where the horizontal x-axis is the Data and the vertical y-axis is the openning price of the stock. Finally, we tell Plotly to give the chart a title.')
    with st.echo():
        diagram = px.line(ticker, x="Date", y="Open", title='Google Stock Openning Prices')
    
    st.write('Show the actual chart:')
    
    __(display=False)
    diagram.show()
    __()
    
    fig = px.line(ticker, x='Date', y='Open')
    st.plotly_chart(fig)
        

    st.header('Get Opening Stock Price for Forecasts')   
    with st.echo():
        data_forecast = ticker[['Date','Open']]
        st.dataframe(data_forecast)     

    st.header('Prepare data for Prophet')
    with st.echo():
        data_forecast.rename(columns={'Date':'ds', 'Open':'y'}, inplace=True)
        data_forecast.sort_values(by='ds', inplace=True)
        st.dataframe(data_forecast)
        

    st.header('Create the model and train Prophet')    
    with st.echo():
        model = Prophet()
        model.fit(data_forecast)
        
    st.header('How far into the future to forecast')    
    with st.echo():
        future = model.make_future_dataframe(periods=365)
        st.dataframe(future)

    st.header('Create the forecasts')
    with st.echo():
        forecast = model.predict(future)
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    st.header('Plot the forecast')
    with st.echo():
        fig1 = model.plot(forecast)
        fig2 = model.plot_components(forecast)

    st.write(fig1)
    st.write(fig2)

    prophet_chart = alt.Chart(ticker).mark_circle().encode(
        x='Date', y='Open', tooltip=['Date', 'Open'])

    st.write(prophet_chart)
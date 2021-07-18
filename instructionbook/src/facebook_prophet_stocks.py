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

    with st.echo():
        def load_dataset(data_link):
            dataset = pd.read_csv(data_link)
            return dataset


    stocks_data = 'https://raw.githubusercontent.com/deusexmagicae/guidebook/main/guidebook/stocks.csv'
    data = load_dataset(stocks_data)

    with st.echo():
        st.dataframe(data)
        fig = px.line(data, x='Date', y='Open')
        st.plotly_chart(fig)

        
    with st.echo():
        data['Date'] =pd.to_datetime(data.Date)
        data.sort_values(by='Date', inplace=True)
        st.dataframe(data)
        fig = px.line(data, x='Date', y='Open')
        st.plotly_chart(fig)
        

    st.header('Get Opening Stock Price for Forecasts')   
    with st.echo():
        data_forecast = data[['Date','Open']]
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

    prophet_chart = alt.Chart(data).mark_circle().encode(
        x='Date', y='Open', tooltip=['Date', 'Open'])

    st.write(prophet_chart)
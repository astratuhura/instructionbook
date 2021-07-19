import streamlit as st
from src.end_code_block import end_code_block as __

def app ():
    st.header('Google Colab Setup')
    st.write('An account is required to use Google Colab')
    st.write('The URL to Google Colab is [Google Colab](https://www.google.com/colab)')
    
    st.markdown('----')
    
    st.header('Your First Python Program in Google Colab')
    st.write('Python programs begins with importing libaries needed to write and run code. For example, the code below placed at the start of the program imports libraries our app will be using:')
    
    with st.echo():
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
    
    st.write('Note the "as" keyword: it means to alias. For example, it is common to alias pandas as pd, and from then on in our code pd will represent pandas, meaning there is no longer a need to type out the entire pandas word.')
        
    st.subheader('Pandas: Python Data Analysis Library')
    st.write('Pandas is a commonly used Python library to manipulate and analyse data.')
    st.write('Pandas could be found here: [link to Pandas](https://pandas.pydata.org/)')
    
    st.subheader('Plotly: A Visual Library for Machine Learning')
    st.write('At times we will be using Plotly to visualise data.')
    st.write('Plotly could be found here: [link to Plotly](https://plotly.com/)')
    
    
    def load_dataset(data_link):
        dataset = pd.read_csv(data_link)
        return dataset
    
    stocks_data = 'https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/stocks.csv'
    ticker = load_dataset(stocks_data)
    
    
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
        fig = px.line(ticker, x="Date", y="Open", title='Google Stock Openning Prices')
    
    st.write('Show the actual chart:')
    
    __(display=False)
    fig.show()
    __()    
    
    
    fig = px.line(ticker, x='Date', y='Open')
    st.plotly_chart(fig)    
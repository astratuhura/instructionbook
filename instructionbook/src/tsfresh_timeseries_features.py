#import utils
import os

from end_code_block import end_code_block as __
import streamlit as st
import pandas as pd
import numpy as np
# from tsfresh import extract_relevant_features
# from tsfresh import extract_features
# from tsfresh import select_features
# from tsfresh.utilities.dataframe_functions import impute

# from tsfresh import defaults
# from tsfresh.feature_extraction import feature_calculators
# from tsfresh.feature_extraction.data import to_tsdata
# from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
# from tsfresh.utilities import profiling
# from tsfresh.utilities.distribution import (
#     ApplyDistributor,
#     DistributorBaseClass,
#     MapDistributor,
#     MultiprocessingDistributor,
# )
# from tsfresh.utilities.string_manipulation import convert_to_output_format

#from sklearn import preprocessing



from src.end_code_block import end_code_block as __

def app():
    st.header('Analysis of Data Features & Attributes with TSFRESH')
    st.write('The URL to TSFRESH is [link to TSFRESH](https://tsfresh.com/)')
    st.write('The official documentation for tsfresh is available from [here](https://tsfresh.readthedocs.io/en/latest/)')
    st.write('An example of a research paper related to tsfresh is available from [this link](https://www.sciencedirect.com/science/article/pii/S0925231218304843)')
    st.write('A short but interesting article on the importance of tsfresh can be found [here](https://towardsdatascience.com/time-series-feature-extraction-on-really-large-data-samples-b732f805ba0e)')
    
    st.markdown('----')
    st.subheader('TSFRESH: what does it do?')
    st.write('Extraction of data features from timeseries is critical to the process of machine learning as it identifies variables that strongly and weakly influence outcomes.')
    st.write('Generally speaking, when we think about time series we often think about forecasting. However, with feature extraction tools like TSFRESH, problem in forecasting could be converted to a prediction problem. Why? Because more data columns will be generated that lends itself to prediction/classification.')
    
    
    st.write("As examples, imagine that following scenarios that contain timeseries data:")
    workshop = ['Stocks (even multiple stocks) - we can forecast future trends as well as predict the future values',
        'Cardiovascular ECG timeseries – predict the signs of an impending heart attack',
        'Manufacturing/machinery - when will a component fail?']
    _container = st.beta_container()
    for idx in workshop:
        _container.write(idx)
        
    st.write('An exhaustive list of the the kinds of features that tsfresh calculates can be found [here](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)')    
    
    st.markdown('----')
    st.header('How to use TSFRESH on time series data')
    st.subheader('Application to the Share of World Diamond Exports Time Series Dataset')
    
    st.write('First install the needed library in Google Colab:')
    st.markdown("""
                ```
                !pip install tsfresh 
                !pip install "featuretools[complete]" 
                !pip install composeml 
                !pip install utils```""")
    
    st.write('Then import all the libraries needed:')
    st.markdown("""
                ```
                import pandas as pd
                import numpy as np
                from tsfresh import extract_relevant_features
                from tsfresh import extract_features
                from tsfresh import select_features
                from tsfresh.utilities.dataframe_functions import impute

                from sklearn import preprocessing

                import composeml as cp
                import featuretools as ft
                import utils
                import os```""")
    
    st.markdown("""
                ```
                # df is short for dataframe
                df = pd.read_csv('/content/drive/MyDrive/MBA/Share-of-Exports-in-Diamonds.csv')
                ```
                """)    
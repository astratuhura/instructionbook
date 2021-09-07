import os
import streamlit as st
from PIL import  Image

# multiple pages handler 
from multipage import multi_page

# import apps from source (src) directory
from src import facebook_prophet_stocks
from src  import welcome
from src import google_colab
from src import tsfresh_timeseries_features
from src import data_preprocess_featuretools
from src import lesson_5_guide
from src import sentiment_analysis
from src import language_translation

# instantiate multipage app handler
app = multi_page()

# main title as app entry point
st.title('MBA509 AI Programming for Business Analytics')
st.title('Instruction Book')
st.write('This instruction book is a step-by-step guide to Programming in AI/Machine Learning in Python.')

# left and right columns
left_col, right_col = st.beta_columns(2)

st.sidebar.header('Navigation to Instruction Books')

# add apps
app.add_page("Welcome to MBA509", welcome.app)
app.add_page("Lesson #2: Google Colab Setup", google_colab.app)
app.add_page("Lesson #3: TSFRESH: Creating Our First Features", tsfresh_timeseries_features.app)
app.add_page("Lesson #3 Extra: Timeseries Forecasting with Facebook Prophet", facebook_prophet_stocks.app)

app.add_page("Lesson #4: Python, Data Preprocessing & Featuretools", data_preprocess_featuretools.app)
app.add_page("Lesson #5: Guide to Predicting Diamond Prices", lesson_5_guide.app)
app.add_page("Lesson #7: Using Deep Learning AI/ML for Sentiments Analysis", sentiment_analysis.app)
app.add_page("Lesson #8: Using Ludwig AI Deep Learning for Learning and Translating Languages", language_translation.app)

# run main app
app.run()
import streamlit as st
import pandas as pd
from src.end_code_block import end_code_block as __

def app():
    st.header('MBA509 Lesson #7: Deep Learning AI/ML for Sentiment Analysis')
    st.write('This lesson will apply Deep Learning to predict user sentiments. We will use Ludwig AI with Twitter dataset to learn and then predict user sentiments.')
    
    st.markdown('----')
    st.write(
        """    
        - STEPS:
            - (1) Go through the usual installation process to install the required libraries. Note: wait until the installation completes and then re-start the Colab runtime using the button displayed in the output messages of the installation process.
            - (2) Import the libraries.
            - (3) Load the Twitter Tweets dataset.
            - (4) Perform some simple exploratory data analysis.
            - (5) Perform data pre-processing.
            - (6) Take small samples from the dataset for training, validation and testing of AI/ML.
            - (7) Perform Machine Learning with Ludwig: https://ludwig-ai.github.io/ludwig-docs/ . Here we will use two key algorithms: Recurrent Neural Networks and BERT (https://huggingface.co/transformers/model_doc/bert.html)
            - (8) Create some tweets and make predictions.   
        """
    )
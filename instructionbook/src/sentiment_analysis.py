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
    
    st.markdown('----')
    st.header('STEP1: Installations')
    st.write('We go through the usual installation processes and remember to wait for it to run to completion, look for the "RESTART RUNTIME" button within the installation process output messages.')
    
    st.markdown("""
                ```
                # install ludwig libraries
                !pip install ludwig
                !pip install ludwig[text]
                !pip install ludwig[visualize]
                !pip install petastorm
                ```  
                """)
    
    st.markdown('----')
    st.header('STEP 2: Import libraries')
    st.write('Once the libraries have been installed, they can now be imported for use.')
    st.markdown("""
                ```
                # import ludwig libraries
                import ludwig
                from ludwig.api import LudwigModel
                from ludwig.visualize import learning_curves, compare_performance, compare_classifiers_predictions
                from ludwig import visualize
                from ludwig.utils.nlp_utils import load_nlp_pipeline, process_text
                from ludwig.utils.data_utils import load_json
                
                # import libraries to create a word cloud
                from wordcloud import WordCloud, STOPWORDS
                
                # import visualisation and core libraries
                from matplotlib import pyplot as plt
                import yaml
                import pandas as pd
                import numpy as np
                
                import logging
                ```  
                """)
    
    st.markdown('----')
    st.header('STEP 3: Load & Pre-process Dataset')
    
    __(display=False)
    # diamonds dataframe created from csv file read and loaded by pandas
    tweets = pd.read_csv('https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/twitter_tweets.csv')
    __()
    
    st.write('The first 10 rows of the tweets.')
    with st.echo():
        # show the tweets dataset
        tweets
        
    st.dataframe(tweets.head(10))
    
    st.write('Show tweets with negative sentiments.')
    
    with st.echo():
        # find negative sentiments
        tweets[tweets['target'].str.match('n')]
        
    st.dataframe(tweets[tweets['target'].str.match('n')])
    
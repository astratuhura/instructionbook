import streamlit as st
import pandas as pd
from PIL import Image
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
    
    st.subheader('Training, Testing and Validation (Sub)-Datasets')
    st.write(
        """
        To prepare for ML, we want to create three smaller datasets.
        What we use is Pandas sampling method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html. Note that we are taking a small fraction of the full Twitter tweets dataset.
        - Sub-datasets:
            - (1) Training: used for teaching ML.
            - (2) Validation: not used to directly train  AI/ML i.e., withheld from training. Used to give an estimate of how well the  model learned during training while also tuning the model's hyperparameters.
            - (3) Test dataset: used to give an unbiased estimate of the model's skill.   
        """
    )
    
    st.markdown("""
                ```
                # take 1% of the data for training
                tweets_train = tweets.sample(frac=0.01, replace=False, random_state=3)
                ```  
                """)
    
    st.markdown('----')
    st.header('STEP 4: Exploratory Data Analysis')
    st.write('The analysis that is performed in this section is the creation of a word cloud.')
    st.markdown("""
                ```
                # use the entire tweets
                processed_train_data = process_text(' '.join(tweets['text']),
                                    load_nlp_pipeline('en'),
                                    filter_punctuation=True,
                                    filter_stopwords=True)
                                    
                # create the word cloud variable to store the words
                wordcloud = WordCloud(background_color='black', collocations=False,
                      stopwords=STOPWORDS).generate(' '.join(processed_train_data))
                
                # use matplotlib to visualise the data stored in wordcloud variable
                plt.figure(figsize=(16,16))
                plt.imshow(wordcloud.recolor(color_func=lambda *args, **kwargs:'white'), interpolation='bilinear')
                plt.axis('off')
                plt.show()                          
                ```  
                """)
    
    word_cloud = Image.open('./src/word_cloud.png')
    st.image(word_cloud, caption='tweeter word cloud')
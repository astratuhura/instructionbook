import streamlit as st
import pandas as pd
from PIL import Image
from src.end_code_block import end_code_block as __

def app():
    st.header('MBA509 Lesson #8: Using Ludwig AI for Language Translations')
    st.write('This lesson will apply Deep Learning to learn English sentences and the equivalent translation in languages such as Portuguese, Hindi and Spanish.')
    
    st.markdown('----')
    st.write(
        """    
        - What we will do:
            - (1) Go through the usual installation process to install the required libraries. Note: wait until the installation completes and then re-start the Colab runtime using the button displayed in the output messages of the installation process.
            - (2) Import the libraries.
            - (3) Load each language dataset.
            - (4) Perform some simple exploratory data analysis.
            - (5) Perform data pre-processing.
            - (6) Take small samples from the dataset for training, validation and testing of AI/ML.
            - (7) Perform Machine Learning with Ludwig: https://ludwig-ai.github.io/ludwig-docs/.
            - (8) Create some English sentences and make predictions.   
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
                import tensorflow_text as tf_text
                ```  
                """)
    
    st.markdown('----')
    st.header('STEP 3: Load & Pre-process Dataset')
    
    __(display=False)
    # diamonds dataframe created from csv file read and loaded by pandas
    hindi = pd.read_csv('https://s3.ap-northeast-1.wasabisys.com/pubdatasets/languages/hin-eng/hin.txt', encoding='utf-8', error_bad_lines=False, sep='\t', header=None)
    __()
    
    st.write('The first 10 rows of the tweets.')
    with st.echo():
        # show the english and hindi dataset
        hindi
        
    st.dataframe(hindi.head(10))
    
    st.write('Create column headings for English-Hindi Language Pairs.')
    
    with st.echo():
        # create column headings
        hindi.columns  = ['english', 'hindi', 'notice']
        
        # remove the unecessary column of data
        hindi.drop('notice', axis=1, inplace=True)
        
    st.dataframe(hindi)
    
    st.subheader('Training Dataset')
    st.write(
        """
        Where there are many rows of data, for this exercise, we want to reduce the number of rows by taking a small random sample of the language pairs.
        What we use is Pandas sampling method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html.
        
        For example, the English-Russian language pairs contain over 430,000 rows, which is too much for demonstration. We reduce this by taking a small fraction of that data.
           
        """
    )
    
    st.markdown("""
                ```
                # take 30% of the data for training
                russian_train = russian.sample(frac=0.30, replace=False, random_state=3)
                ```  
                """)
    
    st.markdown('----')
    
    
    # word_cloud = Image.open('word_cloud.png')
    # st.image(word_cloud, caption='tweeter word cloud')
    
    st.markdown('----')
    st.header('Machine Learning with Ludwig AI')
    
    st.write('Ludwig AI (from Uber) is a platform built ontop of Tensorflow & Keras (from Google).')
    st.write('Ludwig provides a simple declarative method for building and using AI/ML. For example, the basic skeleton of using Ludwig  is given below:')
    
    st.markdown("""
        ```
        # import ludwig
        from ludwig.api import LudwigModel

        # create instructions for building model
        config = {...}
        
        # build the model
        model = LudwigModel(config)
        
        # tain the model
        training_statistics, preprocessed_data, output_directory = model.train(dataset=dataset_file_path)
        
        # or
        training_statistics, preprocessed_data, output_directory = model.train(dataset=dataframe)
        ```
        """)
        
    
    st.write('The essential part is the construction of the instruction that Ludwig needs as a configuration:')
    
    st.header('STEP 4: Prepare Ludwig AI')
    st.write('The example instruction for Ludwig is for English-to-Hindi:')
    
    with st.echo():
        # ludwig requires a configuration setting in order to create the AI model
                hindi_config = {
                    'input_features': [{ 
                        'name': 'english',
                        'type': 'text', 
                        'level': 'word', 
                        'encoder': 'rnn',
                        'cell_type': 'lstm',
                        'reduce_output': None,
                        'preprocessing':{
                            'word_tokenizer':'english_tokenize'}
                    }],
                    'output_features': [{
                        'name': 'hindi', 
                        'type': 'text',
                        'level': 'word',
                        'decoder': 'generator',
                        'cell_type': 'lstm',
                        'attention': 'bahdanau',
                        'reduce_input': None,
                        'preprocessing':{
                            'word_tokenizer':'multi_tokenize'
                        } ,
                        'loss':{
                            'type': 'sampled_softmax_cross_entropy'}
                        }],
                    'training': {
                        'batch_size': 96,
                        'epochs': 5
                    }
                }
        
    st.write('More information on Ludwig Programmatic API: https://ludwig-ai.github.io/ludwig-docs/user_guide/programmatic_api/')
    st.write('And the very important instructions on the creation of the configuration: https://ludwig-ai.github.io/ludwig-docs/user_guide/configuration/ ')
    
    st.header('Important Notes on Machine Learning')
    st.write(
        """
        - Notes:
            - (1) Epochs: one epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. The higher the number of epochs the more the AI/ML will learn and have a chance to increase its accuracy. However, increasing the number of epochs also increase the demand for time and compute resources.
            - (2) Learning rate: when AI/ML searches for an optimal solution, the search space resembles hills and valleys. What is desirable is to find, by descending down a gradient of a hill, towards the lowest valley, which indicates the minimum "loss". This is essentially what learning rate indicates (sometimes called the step-size). Too low a learning rate - the process of walking down the towards the valley takes too long; too high and the learning process will be much more unstable.
            - (3) For more information on learning rate: https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate
        """
    )
    
    st.header('STEP 5: Train Ludwig AI')
    st.write('The example instruction for Ludwig is for English-to-Hindi:')
    st.markdown("""
        ```
        print("Training Model...")
        hindi_train_stats, _, _  = hindi_model.train(dataset=hindi)
        ```
        """)
    
    st.header('STEP 6: Translate From English to Another Language')
    st.write('The example instruction for Ludwig is for English-to-Hindi.')
    
    with st.echo():
        english_idioms = ['The best of both worlds',
                'Speak of the devil',
                'See eye to eye',
                'Once in a blue moon',
                'When pigs fly',
                'To cost an arm and a leg',
                'A piece of cake',
                'Let the cat out of the bag',
                'To feel under the weather',
                'To kill two birds with one stone',
                'To add insult to injury',
                'You can’t judge a book by its cover',
                'Break a leg',
                'A blessing in disguise',
                'Getting a taste of your own medicine']
        
        # convert idioms to dataframe
        idioms_df = pd.DataFrame({'english': english_idioms})   
        
    st.markdown("""
        ```
        # make translations
        predictions_hindi, _ = hindi_model.predict(dataset=idioms_df)
        ```
        """)
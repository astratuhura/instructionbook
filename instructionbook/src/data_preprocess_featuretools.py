import streamlit as st
import pandas as pd
from src.end_code_block import end_code_block as __


def app():
    st.header('Python Programming Fundamentals')
    st.write('This notebook contains examples and further explanations for elementary Python code and related third-party libraries.')
    
    
    st.markdown('----')
    st.header('Installing libraries or modules')
    st.write('Python programs begins with importing libaries needed to write and run code. For example, the code below placed at the start of the program imports libraries our app will be using:')
    
    st.markdown("""
                ```
                !pip install tsfresh
                !pip install "featuretools[complete]"
                !pip install composeml
                !pip install utils
                ```""")
    
    st.write('Pip is a common Python helper that manages installation and dependency resolution for Python apps, libraries and modules. What this means is that pip will install all the other libraries and correct versions that the primary library depends on.')
    
    st.write('On your local PC, the command is simply:')
    st.markdown("""
                ```
                pip install name_of_library
                ```""")
    
    st.write('However, on an online environment, the pip command is prefixed with an ! exclamation mark.')
    st.markdown("""
                ```
                !pip install name_of_library
                ```""")
    
    st.write('Once installed, it is now possible to import the libraries or modules for use:')
    
    st.markdown("""
                ```
                import pandas as pd  # pd is an alias for Pandas
                import numpy as np
                from tsfresh import extract_relevant_features
                from tsfresh import extract_features
                from tsfresh import select_features
                from tsfresh.utilities.dataframe_functions import impute
                ```""")
    
    
    st.markdown('----')
    st.header('Reading data & creating a dataframe')
    st.write('Pandas is the standard tool used to load, read and create dataframes: [link to Pandas Official Site](https://pandas.pydata.org). The standard was to do this is demonstrated by the code below:')
    
    st.markdown("""
                ```
                name_of_data_frame = pd.read_csv('/path/to/csv/file.csv')
                ```""")
    
    st.write('It is up to you to create a descriptive name (name_of_data_frame) for the dataframe that will be created by Pandas. Usually, you will see the use of the variable name df:')
    
    st.markdown("""
                ```
                df = pd.read_csv('/path/to/csv/file.csv')
                ```""")
    
    st.write('More information and examples of what Pandas provides: [link to Pandas documentation](https://pandas.pydata.org/docs/reference/index.html).')
    
    
    st.markdown('----')
    st.header('Handling date and time data')
    
    with st.echo():
        import pandas as pd
    
    def load_dataset(data_link):
        dataset = pd.read_csv(data_link)
        return dataset
    
    
    st.write('Suppose we have an Airbnb a dataframe.')
    
    #  airbnb = 'https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/listings-detailed.csv'
    
    airbnb = 'https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/listings-detailed.csv'
    
    listings = load_dataset(airbnb)
    
    __(display=False)
    # airbnb listing dataframe created from csv file read and loaded by pandas
    listings = pd.read_csv('https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/listings-detailed.csv')
    __()
    
    st.write('Show the first 5 rows using the "head" method available from Pandas:')
    
    __(display=False)
    listings.head()
    __()
    
    st.dataframe(listings.head())
    
    st.write('The "host_since" column was (in the original data) represented as a string or text type. To ensure that the dates are a datetime object, use Pandas:')
    
    __(display=False)
    # use to_datetime on the host_since column of the dataframe to ensure data is a true datetime object and not simply textual data
    pd.to_datetime(listings['host_since'])
    __()
    
    st.markdown('----')
    st.header('Handling percentages data')
    
    st.write('the column to change is host_response_rate and to modify it is simply writing the column_name = column_name (dot) routine or function to apply.')
    __(display=False)
    
    # replace the x% text data in each row (of type str) to type float (decimals) and divide by 100
    listings['host_response_rate'] = listings['host_response_rate'].str.replace(r'%', r'.0').astype('float') / 100.0
    __()
    
    st.markdown('----')
    st.header('Handling dollar signs')
    __(display=False)
    # replace each of the  $ dollar sign to a decimal number (of type float)
    listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
    __()
    
    
    st.markdown('----')
    st.header('Strategies for handling missing values')
    st.write('Below are various strategies used to handle missing values showing up as NaN (not a number) in each columns and rows of the dataframe.')
    __(display=False)
    
    # use the fillna('replacement', inplace=True) function to replace where the host_reponse_time has NaN
    # notice that a few days or more is the worst response rate  - you many select a faster response rate that exists in the data column
    listings['host_response_time'].fillna('a few days or more', inplace=True)
    
    # fill missing values in the host_response_rate with the mean value from the column
    listings['host_response_rate'].fillna(listings.host_response_rate.mean(), inplace=True)
    
    __()
    
    
    st.markdown('----')
    st.header('Python lambda functions')
    st.write('Lambda functions is simply a routine without a name - officially called an anonymous function. For example, the code below tells Python to replace the "f" (false) found in each row with a zero (0); otherwise if it encounters anything other than the letter "f", it is to replace it with the number 1 (true). ')
    __(display=False)
    listings['host_is_superhost'] = listings['host_is_superhost'].apply(lambda x: 0 if x=='f' else 1)
    __()
    
    st.write('So why use a lambda function? Take a look at the example below. This adds 1 to any number x:')
    __(display=False)
    # this creates a function or routine to add 1 to any number x
    def add_one_to_number(x):
        return x + 1
    __()
    
    st.write('Note that functions or routines in Python is defined by writing the keyword "def" infront of the name of a function that we want create. The name of the function is then followed by the colon ":". Also note that the next line is indented to the right. Once the function has been  define, it can be used.')
    
    __(display=False)
    # pass the number 2 to the function defined above
    value = add_one_to_number(2)  
    __()
    
    st.write('Now take a look at a lambda equivalent (below). Notice that the function has not been given a name - hence why it is called a lambda or anonymous function.')
    __(display=False)
    
    # this adds 1 to any number
    lambda x: x+1
    
    # let x=2
    (lambda x: x + 1)(2)
    
    __()
    
    st.markdown('----')
    st.header('Grouping values')
    st.write('To group data by a column, simply use the "groupby" function proivded by Pandas. For example, suppose that we wanted to group by the column "first_review" from the listings dataframe.')
    __(display=False)
    
    # create a variable called listings_groupby_first_review to store the data
    # created by the groupby function
    # groupby will order the dataframe according the the grouping of the first_review column
    listings_groupby_first_review =  listings.groupby('first_review')
    
    __()
    st.write('More information and examples on groupby can be found at the [offcial Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)')
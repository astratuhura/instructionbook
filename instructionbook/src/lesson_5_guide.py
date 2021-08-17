import streamlit as st
import pandas as pd
from src.end_code_block import end_code_block as __
from sklearn.preprocessing import LabelEncoder


def app():
    st.header('Machine Learning for Predicting the Prices of Diamonds')
    st.write('Lesson #5 has two accompanying Python notebooks. The first notebook performs basic data pre-processing, generating new features, and then introduces machine learning. The second notebook performs')
    
    st.write('The second notebook uses the features generated to demonstrates the basics of automated machine learning.')
    
    
    st.markdown('----')
    st.header('The First Notebook')
    st.write('Notebook one, has three main sections: (1) data preprocessing, (2) feature generation, and (3) basic machine learning.')
    
    
    
    with st.echo():
        import pandas as pd
    
    def load_dataset(data_link):
        dataset = pd.read_csv(data_link)
        return dataset
    
    
    st.write('The diamonds dataset.')
    
    #  airbnb = 'https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/listings-detailed.csv'
    
    diamonds = 'https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/diamonds_class.csv'
    
    diamonds = load_dataset(diamonds)
    
    __(display=False)
    # diamonds dataframe created from csv file read and loaded by pandas
    diamonds = pd.read_csv('https://raw.githubusercontent.com/deusexmagicae/instructionbook/main/instructionbook/data/diamonds_class.csv')
    __()
    
    st.write('Show the first 5 rows using the "head" method available from Pandas:')
    
    __(display=False)
    diamonds.head()
    __()
    
    st.dataframe(diamonds.head())
    
    st.write('Remove the Unnamed column.')
    
    diamonds.drop('Unnamed: 0',axis=1,inplace=True)
    
    st.dataframe(diamonds)
    
    st.header('Handle nonsensical data')
    st.write('Notice that every diamond must have a shape i.e., having dimensions x,y, and z. Therefore, it does not make sense to have anyone one (or more) of these values that are zeros.')
    st.write('With this line of code, we want to know where any of x,y, or z are zero. It uses a combination of Pandas .loc[] and Python logical OR operator.')
    
    st.write('The official documentation for [](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html). Pandas .loc[] simply tells us where the condition we are interested in is true.')
    
    with st.echo():
        # which columns x, y and z have zero values
        # the vertical bar is Python's OR logical operator
        zero_values =  diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]
        
        
        st.dataframe(zero_values)
    
    st.write('The strategy used for handling zero values is the use the mean value of each column.')
    st.write('For this purpose we use Pandas [fillna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html). Pandas [mask()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mask.html) method below simply tells us where the values are zero (in this case since we are looking for zero values.') 
    
    st.write('For example, filling the x column with the mean of all the rows of x:')
    
    with st.echo():
        diamonds['x'] = diamonds['x'].mask(diamonds['x']==0).fillna(diamonds['x'].mean())       
        
    
    st.header('Handling categorical data - conversions')
    st.write('For machine learning, we must convert categorical data to its numerical equivalent.')
    
    st.write('First, what are the categorical values?')
    with st.echo():
        # for each column (col) in the cut, color and clarity, show the unique values
        for col in ['cut','color','clarity']:
            print('{} : {}'.format(col,diamonds[col].unique()))
    
    st.write('We can use scikit-learn labelencoder to help us convert from categorical data to numerical data.')
    
    with st.echo():
        # using LabelEncoder of scikit learn
        label_cut = LabelEncoder()
        label_color = LabelEncoder()
        label_clarity = LabelEncoder()

        # now transform  the data in each column of cut, color and clarity into numerical equivalent
        diamonds['cut'] = label_cut.fit_transform(diamonds['cut'])
        diamonds['color'] = label_color.fit_transform(diamonds['color'])
        diamonds['clarity'] = label_clarity.fit_transform(diamonds['clarity'])        
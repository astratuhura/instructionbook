import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import altair as alt

import base64
from io import BytesIO

from end_code_block import end_code_block as __

# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data

# def get_table_download_link(df):
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     val = to_excel(df)
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc

# df = ... # your dataframe
# st.markdown(get_table_download_link(df), unsafe_allow_html=True)

# PS: pip install xlsxwriter  # pandas need this

st.title('The Instruction Book for MBA509')
st.markdown('----')
st.header("Hi, this is our interactive guide book for ai programming in business analytics")
st.markdown('----')
st.header("Title")
st.subheader("But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness")

st.markdown('----')
st.header("History of  Artificial Intelligence")
history  = ['Alan Turing in “Intelligent Machinery” created the definition of Turing Machines and Computation (1936)', 'Warren McCulloch and Walter Pitts, inspired by the discovery of neurons in the brain, created the McCulloch-Pitts artificial neuron (perceptron) (1943) Inspired by McCulloch-Pitts artificial neuron, Marvin Minsky of Princeton University created the first neural network machine called SNARC (Stochastic Neural Analog Reinforcement Calculator)', 'Norbert Wiener: Cybernetics: or Control and communication in the animal and the machine” (1948)','Claude Shannon established Information Theory with “A Mathematical Theory of Communications” (1948)','Isaac Asimov publishes the science-fiction “I, Robot” (1950)', 'Alan Turing  created the Turing Test  in “Computing Machinery and Intelligence” (1950)']
_container = st.beta_container()
for idx in history:
    _container.write(idx)


st.subheader("Dartmouth Summer Research Project on Artificial Intelligence of 1956")
st.write("“every aspect of learning or any other feature of intelligence can be so precisely described that a machine can be made to simulate it”")
workshop = ['Marvin Minsky – creator of SNARC',
'Claud Shannon – Information Theory',
'John McCarthy (who coined the term ”artificial intelligence” to avoid association with cybernetics',
'Herbert A. Simon (1978 Nobel Prize in Economics)',
'Allen Newell – “The Chess Machine: An Example of Dealing with a Complex Task by Adaptation” (1955)',
'Nathaniel Rochester, chief architect of the IBM 701 – the first mass produced scientific computer and the first commercial variant IBM 702, inventor of assembler (machine code)',
'Ray Solomonoff  - Algorithmic Information Theory',
'Arthur Samuel – popularise the term “Machine Learning” (1959)',
'Allen Newell',
'Oliver Selfridge']
_container = st.beta_container()
for idx in workshop:
    _container.write(idx)

st.markdown('----')
st.subheader("The Three Phases of AI")
st.write("Artificial Intelligence is an attempt to replicate or simulate human intelligence, and has three broad phases:")
phases_ai = ['Phase I: Artificial Narrow Intelligence  - designed to perform specific predefined functions (simulates human behaviour',
'Phase II: Artificial General Intelligence – machines posses the ability to reason and make decisions at the level of human intelligence',
'Phase III: Artificial Superintelligence – AI singularity more capable than human intelligence']
_container = st.beta_container()
for idx in phases_ai:
    _container.write(idx)

st.markdown('----')
st.subheader("The Four Types of AI")
st.write("Artificial Intelligence has four main types:")
kinds_ai = ['Reactive Machines  Operates based on the existing data and considers only the current situation Can not perform inferences on the data to evaluate future actions Performs pre-defined functions i.e., Narrow AI',
'Limited Memory: Makes decisions based on past data from its short-term memory; Can evaluate future actions e.g., self-driving cars',
'Theory of Mind: Emotional intelligence – at the research stage', 'Self-aware AI Superintelligence']
_container = st.beta_container()
for idx in kinds_ai:
    _container.write(idx)


st.markdown('----')
st.subheader("Artificial Narrow Intelligence")
arts = ['Rankbrain - Google Search',
'Siri by Apple, Alexa by Amazon, Cortana by Microsoft and others',
'Image and facial recognition','Manufacturing robots and drones',
'Email spam filters','social media monitoring tools for filtered content',
'Recommendation systems based on user viewing preferences (YouTube,  Netflix), listening behaviour (YouTubbe Music, Apple Music, Spotify) and buying patterns (Amazon)',
'Self-driving or autonomous vehicles']
_container = st.beta_container()
for idx in arts:
    _container.write(idx)

st.markdown('----')
st.header("Machine Learning")
texts  = ['Machine Learning are narrow artificial intelligence classes of algorithms designed to for a sole purpose.', 'ML algorithms do not have human-level intelligence, but because of the extraordinary availability of computing power and real-time Big [Business] Data, these algorithms can learn to easily solve business problems.']

select_container = st.beta_container()
select_container.write(texts[0])
select_container.write(texts[1])

# elements of algorithms
st.header("Elements of an Algorithm")
elements =  ['Input: data', 'Computation: a way of performing some operation e.g., addition', 'Selection: a way of selecting among two or more possibilities or paths', 'Iteration: progression or repetition of a set of instructions for a fixed number of steps, or until some logical condition holds ', 'Termination: the sequences of steps (the algorithm) must terminate', 'Output: the solution must be produced']
_container = st.beta_container()
for idx in elements:
    _container.write(idx)


# elements of algorithms
st.header("Algorithms and Problems")
problems =  ['For each different problems or classes of problems, there may be different types algorithms that could solve, or equivalently, decide that the problem has some form of solution(s). For example, there may be different recipes for cooking the same dish', 'There may also be different ways to implement the algorithm', 'An algorithm must solve the problem, or decide if the problem has a solution (is solvable) in a finite amount of time and space', 'Therefore, different algorithms will take different amounts of time and space (memory) to solve problems']
_container = st.beta_container()
for idx in problems:
    _container.write(idx)
    
# elements of algorithms
st.header("Classes of Algorithms")
st.subheader("Deterministic Algorithms")
determine =  ['Deterministic Algorithms: is a type of algorithm that, given some input, will always produce the same output e.g. adding two numbers']
_container = st.beta_container()
for idx in determine:
    _container.write(idx)    

st.subheader("Non-deterministic Algorithms")
nondetermine = ['Non-deterministic Algorithms: is a type of algorithm that, given the same input, the next step in the computation as well as the results, will be un-predictable', 'The next step in the sequence of computation can have two or more possibilities. That is, the path that the algorithm takes is random or in other  words, the algorithm can branch into multiple paths of execution and it is permitted to do so in parallel (e.g., randomly select from a parallel universe)']
_container = st.beta_container()
for idx in nondetermine:
    _container.write(idx)  

st.markdown('----')
# elements of algorithms
st.header("Algorithms in Business Analytics")
algoba =  ['Business Analytics relies on the use of algorithms to solve problems e.g., Machine Learning (algorithms)', 'Why is this important? To do with definition: An algorithm is a finite sequence of well-defined computable instructions', 'The sequences of operations (the steps of the algorithm) must turn the input data into output and this sequence must be correct', 'Question: given the business problem and the input data, are the steps taken to find a solution correct? Will it lead to a useful result? Will it provide insight for decision making?']
_container = st.beta_container()
for idx in algoba:
    _container.write(idx)
    
st.subheader("Input Data")
in_data =  ['The data to be transformed by the  algorithm', 'What data is needed to solve the business problem?', 'How much data?', 'What form of data?']
_container = st.beta_container()
for idx in in_data:
    _container.write(idx)
st.markdown('----')

st.header('Types of Machine Learning')
st.write("""
Using algorithms to solve business problems through data processing, interpretation and analysis, and has the following sub-categories:
- Supervised Learning: learns to map input data to target variables
- Unsupervised Learning: describes and extract relationships in data
- Semi-supervised Learning: a combination of both supervised and unsupervised learning algorithms
- Reinforcement Learning: interacts with the environment
- Deep Learning:
""")



st.markdown('----')
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
    @st.cache
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
    data.sort_values(by='Date', inplace=True) # This now sorts in date order
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

st.markdown('----')
path = st.text_input('CSV file path')
if path:
    df = pd.read_csv(path)
    df

st.markdown('----')
st.write("""
[Streamlit](https://streamlit.io/) is [announced](https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace) as being **The fastest way to build custom Machine Learning tools** but I believe it has the potential to become much more awesome than that.
I believe Streamlit has the **potential to become the Iphone of Data Science Apps**. And maybe it can even become the Iphone of Technical Writing, Code, Micro Apps and Python.
The **purpose** of the **Awesome Streamlit Project** is to share knowledge on how Awesome Streamlit is and can become.
This application provides
- A list of awesome Streamlit **resources**.
- A **gallery** of awesome streamlit applications.
- A **vision** on how awesome Streamlit can be.
## The Magic of Streamlit
The only way to truly understand how magical Streamlit is to play around with it.
But if you need to be convinced first, then here is the **4 minute introduction** to Streamlit!
Afterwards you can explore examples in the Gallery and go to the [Streamlit docs](https://streamlit.io/docs/) to get started.
""")

st.markdown('----')
st.title('Display some data')
df = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

st.write(c)

st.markdown('----')
st.button('Click here')
st.checkbox('Check')
st.radio('Radio', [1,2,3])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiple selection', [21,85,53])
st.slider('Slide', min_value=10, max_value=20)
st.select_slider('Slide to select', options=[1,2,3,4])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Text area')
st.date_input('Date input')
st.time_input('Time input')
st.file_uploader('File uploader')
st.color_picker('Color Picker')    
    
st.markdown('----')
st.header("Miscellaneous widgets")
st.markdown("Bold: this is a __bold__ parameter")
st.markdown("Italic: this is a _italic_ parameter")
st.markdown("Bold italic: this is a __*bold italic*__ parameter")
st.markdown("weblink: [Google](https://www.google.com)")

st.markdown('A section')
st.write('Lorem ipsum widgets')
st.markdown('----')
st.subheader('Lorem ipsum widgets')
st.write('Lorem ipsum widgets')
st.markdown('----')

texts  = ['text 1', 'text 2', 'text 3']
st.subheader("What is the difference between")
st.write(texts[0])
st.write(texts[1])
st.write(texts[2])

select_container = st.beta_container()
select_container.write(texts[0])
select_container.write(texts[1])
select_container.write(texts[2])

title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

genre = st.radio(
"What's your favorite movie genre",
('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")

agree = st.checkbox('I agree')

if agree:
    st.write('Great!')

left_column, right_column = st.beta_columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

option = st.selectbox(
'How would you like to be contacted?',
('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)


options = st.multiselect(
'What are your favorite colors',
['Green', 'Yellow', 'Red', 'Blue'],
['Yellow', 'Red'])

st.write('You selected:', options)

with st.beta_container():
    options_in = st.multiselect(
    'Select which applies',
    ['Finite', 'Infinite', 'Verifiable', 'Terminate'],
    ['Finite', 'Verifiable'])

    st.write('You selected:', options_in)

# You can call any Streamlit command, including custom components:
st.bar_chart(np.random.randn(50, 3))

st.write("This is outside the container")

st.write('Select three known variables:')
option_s = st.checkbox('displacement (s)')
option_u = st.checkbox('initial velocity (u)')
option_v = st.checkbox('final velocity (v)')
option_a = st.checkbox('acceleration (a)')
option_t = st.checkbox('time (t)')
known_variables = option_s + option_u + option_v + option_a + option_t

if known_variables <3:
    st.write('You have to select minimum 3 variables.')
elif known_variables == 3:
   st.write('Now put the values of your selected variables in SI units.')
elif known_variables >3:
    st.write('You can select maximum 3 variables.')


df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.markdown(df.index.tolist())



chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")
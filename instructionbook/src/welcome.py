import streamlit as st

def app():
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
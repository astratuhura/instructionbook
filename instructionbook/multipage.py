# manages different Python apps for streamlit

import streamlit as st


class multi_page: 
    

    def __init__(self) -> None:
        # each app is a page to navigate to
        self.pages = []
    
    # add an app as a streamlit page
    def add_page(self, title, func) -> None: 
        '''Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        '''

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        
        # the navigation sidebar  
        page = st.sidebar.radio(
            # navigation title
            'Instruction Books', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        # run the app function 
        page['function']()

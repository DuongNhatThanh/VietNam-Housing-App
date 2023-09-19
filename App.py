import streamlit as st
from streamlit_option_menu import option_menu
from screens.index import get_routes

st.set_page_config(
    layout="wide",
)

routes = get_routes()

with st.sidebar:
    global selected_screen
    
    selected_screen = option_menu("Taskbar", routes['name'], 
        icons=routes['icon'], menu_icon="book")
    
    # selected_screen = on_hover_tabs(tabName=routes['name'], 
    #                      iconName=routes['icon'], default_choice=0)

routes['component'][selected_screen]()



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


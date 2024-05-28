import streamlit as st
from streamlit_option_menu import option_menu

import detection, home

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Detection"],  # required
                icons=["house", "search-heart", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                styles={
                "nav-link": {
                    "--hover-color": "#F8D7D7",
                },
                "nav-link-selected": {"background-color": "#DE2165"},
                },
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Detection", "Contact"],  # required
            icons=["house", "search-heart", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Detection", "Contact"],  # required
            icons=["house", "search-heart", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#DE2165",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=1)

if selected == "Home":
    home.app()

if selected == "Detection":
    detection.app()

#if selected == "Contact":
    #st.title(f"You have selected {selected}")
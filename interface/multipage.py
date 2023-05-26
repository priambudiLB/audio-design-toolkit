import streamlit as st


class MultiPage:
    def __init__(self):
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
        self.apps = []
        self.app_names = []

    def add_app(self, title, func, args=None):
        self.app_names.append(title)
        self.apps.append({
            "title": title,
            "function": func,
            "args":args
        })

    def run(self):
        # get query_params
        query_params = st.experimental_get_query_params()
        choice = query_params["app"][0] if "app" in query_params else None
        # common key
        for app in self.apps:
            if app['title'] == choice:
                app['function']()
from multipage import MultiPage
from perceptual_guided_control import main as pg_control_main
from query_gan import main as query_gan_main
from query_gan_react import main as query_gan_react
from sefa import main as sefa_main


import streamlit as st

# Based on and modified from https://github.com/upraneelnihar/streamlit-multiapps

app = MultiPage()
app.add_app('our-algo', pg_control_main)
app.add_app('query-gan', query_gan_main)
app.add_app('query-gan-react', query_gan_react)
app.add_app('sefa-algo', sefa_main)
app.run()
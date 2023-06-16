import streamlit as st
import pandas as pd
import plotly.express as px
from intro import intro_tab
from analysis import analysis_tab
from implementation import implementation_tab
from tryit import tryityourself_tab
from chatgpt import chatgpt_tab

# st.set_page_config(layout="wide")

st.title("Media Bias x ChatGPT")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Analysis", "Implementation", "Is ChatGPT politically biased?", "Try it yourself"])

with tab1:
    intro_tab()

with tab2:
    analysis_tab()

with tab3:
    implementation_tab()

with tab4:
    chatgpt_tab()

with tab5:
    tryityourself_tab()

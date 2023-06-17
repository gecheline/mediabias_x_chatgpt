import streamlit as st
import pandas as pd
import plotly.express as px
from intro import intro_tab
from analysis import analysis_tab
from chatgpt import chatgpt_tab

# st.set_page_config(layout="wide")

st.title("Media Bias x ChatGPT")

tab1, tab2, tab3= st.tabs(["Introduction", "Analysis", "Is ChatGPT politically biased?"])

with tab1:
    intro_tab()

with tab2:
    analysis_tab()

with tab3:
    chatgpt_tab()


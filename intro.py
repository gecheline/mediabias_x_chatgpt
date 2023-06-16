import streamlit as st
import pandas as pd
import plotly.express as px

def dataset_explorer():
    """Plots a histogram of the specified column in the dataframe."""
    df = pd.read_excel('data/final_labels_SG2.xlsx')
    col1, col2 = st.columns([1,3])
    with col1:

        columns = ['topic', 'outlet', 'type', 'label_bias', 'label_opinion']
        column = st.selectbox('Select Column to visualize:', columns, index=3)

        # st.write("Select a color column and barmode:")

        color = st.selectbox('Split bars based on additional column:', ['None'] + [c for c in columns if c != column], index=0)
        if color == "None":
            color = None

        barmode = st.radio('Barmode', ['group', 'stack'], index=0)

    with col2:
        # Display the histogram
        hist_fig = px.histogram(df, x=column, color=color, barmode=barmode)
        st.plotly_chart(hist_fig)


def intro_tab():
    st.header("The question")
    st.write("This app serves to explore whether Large Language Models (LLMs), in particular GPT-3.5, are biased on topics that have been largely covered in American news outlets in the past years. To tackle this problem, we'll train a ML model that classifies sentences into biased and non-biased based on data sourced from various media outlets and labeled by human experts.")

    st.header("The data")
    st.write("This app explores data from the [Media Bias Detection group](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE/tree/main/data). The data also contains media outlet political bias evaluations from [AllSlides](https://www.allsides.com/media-bias/media-bias-chart) and is complemented by data from [Media Bias / Fact Check](https://mediabiasfactcheck.com/).")
    st.write("*This data set has also been used in the paper 'Neural Media Bias Detection Using Distant Supervision With BABE - Bias Annotations By Experts' by T.Spinde et al. for development of a bias detection algorithm based on a pre-trained BERT model on headlines from various outlets and labels based on the outlet political bias. Their models will only be used here for comparison on the performance of my ML pipeline.*")
    st.write("For a deeper look into the differences between BERT (used in published work) and ADA (used in this work) models, see this [post](https://blog.invgate.com/gpt-3-vs-bert).")

    st.subheader("Explore the dataset")
    dataset_explorer()


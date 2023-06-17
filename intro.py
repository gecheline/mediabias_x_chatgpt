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
    st.header("Introduction")
    # st.write("This app serves to explore whether Large Language Models (LLMs), in particular GPT-3.5, are biased on topics that have been largely covered in American news outlets in the past years. To tackle this problem, we'll train a ML model that classifies sentences into biased and non-biased based on data sourced from various media outlets and labeled by human experts.")
    st.write('''
    This project was inspired by my own interactions with OpenAI's ChatGPT interface and an interesting question they have raised for me through the process: "Is ChatGPT perceived as politically biased within the context of (predominantly American) news reporting?". For example, asking ChatGPT to write a fairytale about a society thriving on fossil fuels, regardless of the prompt, leads to some version of a clean energy outcome. Here are snippets of one such story it generated for me:

> *Once upon a time, in a land far away, there existed a kingdom named Petrovia. This extraordinary kingdom thrived on the power of fossil fuels, which brought prosperity to its people. The land was abundant with coal mines, oil wells, and natural gas reserves that fueled the kingdom's progress.*

> *The king of Petrovia, King Enerius, ruled with wisdom and foresight. He knew that the kingdom's reliance on fossil fuels brought wealth and comfort to his people, but he also understood the importance of balance and sustainability. King Enerius set forth a decree that dictated the responsible extraction and consumption of these precious resources, ensuring the well-being of Petrovia's natural environment.*

> *However, there was a young orphan named Lila who lived on the outskirts of the kingdom. She had a deep connection with nature and possessed a keen sense of curiosity. Lila often wondered if there was another way for Petrovia to thrive without depleting its fossil fuel reserves and harming the environment.*

> *The tale of Petrovia became known far and wide as a story of enlightenment and progress. Other kingdoms, inspired by the kingdom's success, began their own journey toward sustainable energy. The world learned that it was possible to thrive without relying solely on finite resources.*

Asking ChatGPT controversial questions also leads to either it refusing to offer a response or providing a balanced one that focuses on facts and evidence rather than speculation, but with my own understanding of reporting in American media, I've come to an assumption that it may be perceived as left-leaning.

To test this assumption, I trained a classification model on the BABE dataset, which contains about ~3500 sentences extracted from articles published by a variety of news outlets: left, center and right. The sentences are labeled by human experts as "Biased" or "Non-biased", as well as "Entirely factual", "Expresses author's opinion" or "Somwhat factual but also opinionated". The final labels are determined from agreement between expert labels, and a few also have the "No agreement" label as a final label in both of these categories. In addition, each sentence has a topic label and the outlets and links to the original articles are provided. Some outlets also have a political bias rating that I'll revise and update.
    ''')
    st.header("The data")
    st.write("This app explores data from the [Media Bias Detection group](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE/tree/main/data). The data also contains media outlet political bias evaluations from [AllSlides](https://www.allsides.com/media-bias/media-bias-chart) and is complemented by data from [Media Bias / Fact Check](https://mediabiasfactcheck.com/).")
    st.write("*This data set has also been used in the paper 'Neural Media Bias Detection Using Distant Supervision With BABE - Bias Annotations By Experts' by T.Spinde et al. for development of a bias detection algorithm based on a pre-trained BERT model on headlines from various outlets and labels based on the outlet political bias. Their models will only be used here for comparison on the performance of my ML pipeline.*")
    st.write("For a deeper look into the differences between BERT (used in published work) and ADA (used in this work) models, see this [post](https://blog.invgate.com/gpt-3-vs-bert).")

    st.subheader("Explore the dataset")
    dataset_explorer()


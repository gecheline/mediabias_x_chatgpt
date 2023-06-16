import streamlit as st
import pandas as pd
import plotly.express as px

def chatgpt_tab():
    st.header("Is ChatGPT politically biased?")
    st.write("In my experience with interacting with ChatGPT I have been pleasantly surprised by how diplomatic and balanced its responses seem, even on controversial topics. For example, I asked it to write me a story about a society thriving on fossil fuels and even though it started with my prompt on fossil fuel thriving, the story's outcome led to the discovery of clean energy sources and protecting the environement.")
    st.write("Exercises like this one led me to suspect it might be percieved as left-leaning within the context of American media, which was the main inspiration for this project.")

    st.subheader("Media Bias Dataset <> ChatGPT")
    st.write("To assess the overall topic-reliability, bias and perceived political bias, I first asked ChatGPT to write a couple of sentences on articles similar to the ones the training data was sourced from.")
    st.write("To achieve this, I parsed the url links in the data and extracted the article tags. Then I sampled 50 of the extrated tags and manually chose 5 from each outlet type (center, left, right) to test with ChatGPT.")
    st.write("The prompt I gave to ChatGPT for each one was formatted as follows:")
    st.code('Can you write two-three sentences on the following topic: [insert parsed article taf] in the style of a news article')
    st.write("I captured ChatGPT's responses in a pandas DataFrame and ran the prediction model for each one.")
    df_gpt = pd.read_csv('data/gpt_responses.csv').drop(columns=['Unnamed: 0', 'embedding'])
    st.write(df_gpt)
    st.write("From the data and charts below we can see that all Chat-GPT generated content was classified as non-biased, which is a good testament to OpenAI's commitment to offering a non-biased perspective (see an interesting conversation I had with ChatGPT while researching this project at the bottom of this page).")
    hist_fig_1 = px.histogram(df_gpt, x='label_bias')
    st.plotly_chart(hist_fig_1)
    st.write("However, in terms of political bias, most of the content is labeled as likely originating from a left-leaning outlet. This serves to potentially validate my initial hunch that ChatGPT generated content may be perceived as left-leaning, but it opens a whole new set of questions.")
    st.markdown('''
    - does this mean ChatGPT is politically biased or does it say something about the style of reporting of left VS right-wing media?
    - being trained on online data with human supervision, how much does said human supervision affect this perceived political bias?
    - is the outlet bias not a good indicator of an individual sentence's political bias?
    - a hot take: is right-wing media so biased that it affects the overall scale so that center media appear left in comparison?
    ''')
    hist_fig_2 = px.histogram(df_gpt, x='outlet_bias')
    st.plotly_chart(hist_fig_2)
    st.write("Answering these questions would take a whole extra project so I'll leave them as parting remarks for now!")

    st.subheader("*Bonus: chatting politics with ChatGPT*")
    st.image('data/chatgpt_politics.png')





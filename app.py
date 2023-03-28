import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
try:
    stop = set(stopwords.words('french'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop = set(stopwords.words('french'))

st.set_page_config(
    page_title="ISJ-MSII-DS",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
dataset_url = "french_tweets.csv"
# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url,nrows=10000)

df = get_data()

# dashboard title
st.title("Projet 2 : Résumé visuel de texte pour le management")

# top-level filters
job_filter = st.selectbox("Choisir le sentiment", pd.unique(df["label"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df = df[df["label"] == job_filter]

# near real-time / live feed simulation
for seconds in range(200):

    # df["age_new"] = df["age"] * np.random.choice(range(1, 5))
    # df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))

    # # creating KPIs
    # avg_age = np.mean(df["age_new"])

    # count_married = int(
    #     df[(df["marital"] == "married")]["marital"].count()
    #     + np.random.choice(range(1, 30))
    # )

    # balance = np.mean(df["balance_new"])

    with placeholder.container():

        # create three columns
        # kpi1, kpi2, kpi3 = st.columns(3)

        # # fill in those three columns with respective metrics or KPIs
        # kpi1.metric(
        #     label="Nombre de ligne",
        #     value=round(avg_age),
        #     delta=round(avg_age) - 10,
        # )
        
        # kpi2.metric(
        #     label="Nombre de mot",
        #     value=int(count_married),
        #     delta=-10 + count_married,
        # )
        
        # kpi3.metric(
        #     label="Moyenne des mots par Tweet",
        #     value=f"$ {round(balance,2)} ",
        #     delta=-round(balance / count_married) * 100,
        # )

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:

            st.markdown("### Histogramme 1:")
            st.markdown("#### Mot par Tweet")           
            fig = px.histogram(data_frame=df['text'].str.len(), x="text")
            st.write(fig)
            
        with fig_col2:
            st.markdown("### Histogramme 2:")
            st.markdown("#### Mot par ligne")   
            fig2 = px.histogram(data_frame=df['text'].str.split().map(lambda x: len(x)), x="text")
            st.write(fig2)


        st.markdown("### Histogramme 3:")
        st.markdown("#### Moyenne des mots par Tweet") 
        fig3 = px.histogram(data_frame=df['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: round(np.mean(x),0)), x="text")
        st.write(fig3)

        corpus=[]
        new= df['text'].str.split()
        new=new.values.tolist()
        corpus=[word for i in new for word in i]

        from collections import defaultdict
        dic=defaultdict(int)
        for word in corpus:
            if word in stop:
                dic[word]+=1

        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
        x,y=zip(*top)

        st.markdown("### Histogramme 4:")
        st.markdown("#### Distribution des stops words") 
        #fig4 = plt.bar(x,y)
        fig4 = px.histogram(x,y)
        st.write(fig4)





        st.markdown("### Detailed Data View")
        st.dataframe(df)
        time.sleep(1)

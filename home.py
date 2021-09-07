import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px

import seaborn as sns
from matplotlib import pyplot as plt

import base64
from PIL import Image

plt.style.use('fivethirtyeight')

def app():
    st.title('About Me!')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/nneuenschwander/) - [GitHub](https://github.com/nneuenschwander) - 770.910.3405')
    st.write("Nicholas Neuenschwander holds a Bachelor's of Science from Kennesaw State University and is currently pursing his Masters in Computer Science from Georgia Tech (Expected Fall 2022). Nicholas lives in the Atlanta area with his spouse and 2 dogs.")
    st.write("Nicholas has spent the past 7 years working in Analytics & Data Science. Recently, he's lead a small team of data scientists focused on Data Stewardship building time-series models for anomaly detection and forecasting algorithms. Nicholas is well versed in Python, SQL, Linux, and Modeling Techniques.")
    st.write("When not focused on work or school, Nicholas spends his time as an avid animal lover. Previously working with dog rescue groups, Nicholas has fostered over 20 different dogs while having 2 of his own: a pug named Bob, and Shepherd named Joe. Nicholas also loves wood and metal working in any free-time he has left.")
    st.image(Image.open('img/meBobJoe.png'))
    displayPDF('nneuenschwander-resume.pdf')

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
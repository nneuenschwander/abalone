import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px

import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

def app():
    st.title('Summary')
    st.write('The abalone problem dealt with data imbalance and how to improve predictions for the minority class. When data sets are highly imbalanced, the majority class will often overwhelm the model and create a learner that likes to predict classification as the majority class. In many cases, this is not ideal as we would like to be able to confidently sort the data for the minority class as well. We can combat data imbalance using different sampling techniques.')
    st.write('Using Over-Sampling and Under-Sampling methods, we were able to achive an f1-score of 0.64 with a recall of 0.80 for the minority class. This is well able the baseline model where we achieved a recall of 0.41 with an f1-score of 0.54 for the minority class.')
    st.markdown('## __Other Considerations__')
    st.markdown(''' 
* Feature Engineering
    * We did not fully explore feature engineering to the fullest potential
    * We were able to use PCA to reduce the feature set
    * With more time, I would have explored using [Featuretools](https://www.featuretools.com/)
        * Featuretools is an open Python framework for automated feature engineering
        * Best used with relational data
    * Model Exploration
        * During our modeling exercise, we tinkered with `LogisticRegression()`
        * I also looked at other tree models (i.e XGBoost, RandomForestClassifier)
        * These models did not perform as well as LogisticRegression and for the sake of brevity were removed from the analysis
    ''')
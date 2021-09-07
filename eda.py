import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px

import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

def app():
    # Data Dictionary from UCI
    st.title('Abalone Data Set EDA')
    st.markdown('Welcome to an exploration of the Abalone data set. The file has 9 columns and 4,178 records on disk. The objective is to predict whether the number of rings will be greater than 5 or not.')
    st.markdown("From [UC Irvine ML Repository](https://archive.ics.uci.edu/ml/datasets/Abalone 'Abalone') we can get a definition of the data set features as follows:")
    st.markdown('### __Table 1: Data Dictionary as Stated from UCI ML Repo__')

    st.markdown('''|Name | Data Type | Measurement Unit | Description|
    |---:|:------|---------:|-----------:|
    | Sex | nominal | -- | M, F, and I (infant) |
    | Length | continuous | mm | Longest shell measurement |
    | Diameter | continuous | mm | perpendicular to length |
    | Height | continuous | mm | with meat in shell |
    | Whole weight   | continuous | grams | whole abalone |
    | Shucked weight | continuous | grams | weight of meat |
    | Viscera weight | continuous | grams | gut weight (after bleeding) |
    | Shell weight | continuous | grams | after being dried |
    | Rings | integer | -- | +1.5 gives the age in years |''')

    df = get_data()
    initial_features = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    st.markdown('### __Table 2: First 5 records of abalone.csv__')
    st.dataframe(df[initial_features].head())
    st.markdown("The abalone.csv data set has 4,177 rows which is consistent with the row count of the file. There are 9 columns, 8 of which are continuous. The column 'sex' is categorical with 3 values [M,F,I]")
    st.markdown('### __Table 3: Descriptive statistics for continuous features__')
    st.dataframe(df[initial_features].describe())
    st.markdown("The minimum height for abalone is 0 which indicates that the data is possibly missing. Inspecting the records, there are only 2 that are missing 'height', we will need to impute these values or remove from data set.")
    st.markdown('### __Table 4: Records where height = 0.0__')
    st.dataframe(df[initial_features][ df.height==0 ])

    figures = 1
    st.markdown('## __Distribution of Continuous Features__')
    continuous_features = [ 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    for cf in continuous_features:
        h = px.histogram(df,x=cf,title=f'Figure {figures}: Histogram of {cf}')
        figures += 1
        b = px.box(df,x=cf,title=f'Figure {figures}: Boxplot of {cf}')
        figures += 1
        st.plotly_chart(h)
        st.plotly_chart(b)

    st.markdown('From the `height` BoxPlot, We can see that height also has 2 larger data points of 0.515 and 1.13 which is much higher than the median of 0.14.')

    fig = sns.pairplot(df[initial_features],corner=True)
    plt.suptitle(f'Figure {figures}: Pairplot of Continuous Features',fontsize=20)
    figures += 1
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df[initial_features].corr(), ax=ax)
    plt.title(f'Figure {figures}: Correlation Heat Map for Continuous Features')
    figures += 1
    st.pyplot(fig)

    st.markdown('### Correlation')
    st.markdown('From the Pair Plot and Heat map, we can see that there is high correlation between the different weight features. This can cause multicolinearity in a model that we may need to account for. We can also see that the length and diameter are highly correlated.')
    st.markdown('### Weights')
    st.markdown("From the corrlelation of the weight features, it follows that the whole weight of an abalone should be greater than or equal to it's components. We can check this in the data by creating a weight delta feature as follows:")
    st.markdown('$weight \Delta = whole weight - (shucked weight - viscera weight - shell weight)$')
    st.markdown('The weight $\Delta$ should be greater than 0 in all cases, as the whole weight should include some discard of the initial abalone not accouned for in the 3 parital weight features. However, we can see that there are records where the weight $\Delta$ is negative.')

    initial_features.append('weight_delta')
    st.markdown('### __Table 5: Descriptive statistics for records with negative weight $\Delta$__')
    st.dataframe(df[ df.weight_delta < 0 ][initial_features].describe())
    st.markdown('### __Table 6: First 5 records with negative weight $\Delta$__')
    st.dataframe(df[ df.weight_delta < 0 ][initial_features].head())
    st.markdown(f'Errors in measurement or data collection could possibly explain the inconsistencies in the weight features. Looking at `Figure {figures}`, we can see that many of the negative weight $\Delta$ values are close to 0 which may be due to rounding errors.')

    fig = sns.displot(df[ df.weight_delta < 0 ].weight_delta)
    plt.title(f'Figure {figures}: Distribution of Negative Weight Δ values')
    figures += 1
    st.pyplot(fig)

    st.markdown('## __Distribution of Categorical Features__')

    # sex variable
    fig, ax = plt.subplots()
    sns.countplot(x=df['sex'],order=df.sex.value_counts().index,ax=ax)
    # add % as label
    for p, label in zip(ax.patches, df['sex'].value_counts()/df.shape[0]):
        ax.annotate(f'{label:.2%}', (p.get_x()+0.25, p.get_height()+0.1))
    plt.title(f'Figure {figures}: Distribution of Sex')
    st.pyplot(fig)
    st.markdown(f'There is a slightly higher representation of males in the data set in relationship to females and infants as seen in `Figure {figures}`')
    figures += 1

    # response
    st.markdown(f'### Creating response variable')
    st.markdown(f"The objective is to predict whether or not an observation will have greater than 5 rings or not. Currently, rings is a continuous variable that we will need to convert into a binary variable that will end up being the response variable of our modeling efforts. `Figure {figures}` shows the distribution of records that have rings greater than 5. As we can see, 95% of the data set has a more than 5 rings, we will have to consider different sampling methods in order to account for this.")

    fig, ax = plt.subplots()
    sns.countplot(x=df.y,order=df.y.value_counts().index,ax=ax)
    for p, label in zip(ax.patches, df['y'].value_counts()/df.shape[0]):
        ax.annotate(f'{label:.2%}', (p.get_x()+0.25, p.get_height()+0.1))
    plt.title(f'Figure {figures}: Distribution of Response\nRings > 5')
    st.pyplot(fig)

    st.markdown('# Summary')
    st.markdown('''
        1. The response variable is highly imbalanced
            * Imbalanced data can overwhelm a model towards the majority class
            * Will need to handle using sampling methods
        2. There are some data issues that we may need to handle
            * Missing height
            * Negative weight $\Delta$
        3. Many of the features are correlated
            * Will have to use methods of feature selection to avoid multicollinearity
    ''')

@st.cache
def get_data():
    df = pd.read_csv('data/abalone.csv')
    df.columns = [ '_'.join(c.lower().split()) for c in df.columns ]
    partial_weights = [ c for c in df.columns if 'weight' in c and 'whole' not in c ]
    df['weight_delta'] = df.whole_weight - df[partial_weights].sum(axis=1)
    df['y'] = (df.rings > 5).astype(np.int16)

    return df
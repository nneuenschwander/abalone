import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,RocCurveDisplay,auc,plot_confusion_matrix,confusion_matrix,f1_score,balanced_accuracy_score,make_scorer

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

rs = 271828

import streamlit as st
from PIL import Image

def app():
    st.title('Modeling')
    st.markdown('## __Creating dummy variables for sex feature__')
    st.write('The abalone data set has one categorical variable that we will need to encode for modeling. One way to do this is with the pandas make_dummies. This will give is binary features for each category in sex, we can also remove the final category the 3rd category is a linear combination of the other two.')
    st.code("df[['infant','male']] = pd.get_dummies(df.sex,drop_first=True)",language='python')
    df = get_data()
    tables = 1
    figures = 1
    st.markdown(f'### Table {tables}: Creating Dummy Variable for sex')
    tables += 1
    st.dataframe(df[['sex','infant','male']].head())
    st.write('We can see that all 3 categories can be represented with the 2 bitwise columns. After creating the new features, we can remove the origial string feature from modeling.')

    st.markdown('## __Train/Test Split__')
    st.write('In order to get an unbiased estimate of model performance, we need to split the data into train and test. We will use the train set to train the model and the test set will be used to see how the model performs against data that the model was not exposed. To accomplish this, we will use the `sklearn` preprocessing package, `train_test_split`.')
    st.write('First, we will create our base X and y vectors using the features we intend to use as modeling. We are removing `rings` from the X set, as it was directly used to create the target variable (y = rings>5).')

    x_features = ['length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','infant','male']
    X = df.loc[:,x_features]
    y = df.loc[:,['y']]

    st.code('''x_features = ['length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','infant','male']
X = df.loc[:,x_features]
y = df.loc[:,['y']]''',language='python')
    st.markdown(f'### Table {tables}: X features')
    st.dataframe(X.head())
    tables += 1
    st.markdown(f'### Table {tables}: y (target variable)')
    tables += 1
    st.dataframe(y.head().T)

    test_pct = 0.3
    rs = 271828
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=rs)

    st.write('Finally we use `test_train_split` to create our final train/test sets.')
    st.code('''test_pct = 0.3
rs = 271828
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=rs)''',language='python')
    st.write(f'X vector shape: {X.shape}')
    st.write(f'y vector shape: {y.shape}')
    st.write('-'*20)
    st.write('Train/Test Split')
    st.write(f'Train pct: {1-test_pct:.2%}')
    st.write(f'Test pct: {test_pct:.2%}')
    st.write('-'*20)
    st.write(f'Train X: {X_train.shape}\tTrain y: {y_train.shape}')
    st.write(f'Test X: {X_test.shape}\tTest y: {y_test.shape}')

    st.markdown('## __Missing Value Imputation__')
    st.write('During EDA, we noticed that the MIN value for the height of an abalone was 0.0, which does make sense. One method of dealing with this is missing value imputation using KNN. This will method will use the other non-missing features to impute the missing data. In order to accomplish this, we will need to convert the bad data into `np.nan` as seen below.')
    st.write('We create an inputer object using the training data set, and use the same object to transform the test set. However, in this case both of the bad height records are in train.')
    st.code('''imputer = KNNImputer(n_neighbors=5)
X_train = pd.DataFrame(imputer.fit_transform(X_train),index=X_train.index,columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test),index=X_test.index,columns=X_test.columns)''',language='python')

    st.markdown(f'### Table {tables}: Records with height=0.0, converted to np.nan')
    st.dataframe(X[ df.height.isna() ])
    tables += 1

    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train),index=X_train.index,columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test),index=X_test.index,columns=X_test.columns)

    st.markdown(f'### Table {tables}: Train records with height=0.0, after missing value imputation')
    st.dataframe(X_train.loc[df.height.isna(),:].sort_index())
    tables+=1
    st.markdown(f'### Table {tables}: Test records with height=0.0, after missing value imputation')
    st.dataframe(X_test.loc[df.height.isna(),:].sort_index())
    tables+=1

    st.markdown('## __Baseline Model__')
    st.write('We can now create a baseline model using our Train/Test sets to see how accurate we are starting out. For Classification problems, there are several models we can choose from, however I will start using a `LogisticRegression()` model from sklearn')
    scores = []
    st.code('''lr_baseline = LogisticRegression()
lr_baseline.fit(X_train.values,y_train.values.ravel())''',language='python')
    st.write('Looking at the AUC in the ROC Curve below, we might conclude that this model is performing well. However, upon closer inspection we can see in the table below that the recall for the negative class is low for both the Train and Test set. This is because during EDA, we noticed that the target variable y (rings>5) is highly imbalanced. We noticed that 95% of the entire data set was part of the positive class, therefore the model is overwhelmed by this and is only able to predict 41% of the negative class correctly in the test set.')
    lr_baseline = LogisticRegression()
    lr_baseline.fit(X_train.values,y_train.values.ravel())
    score,figures = get_classification_metrics(lr_baseline,X_train.values,y_train.values.ravel(),X_test.values,y_test.values.ravel(),model_name='LR Baseline',figures=figures)
    scores.append(score)    
    st.markdown(f'### Table {tables}: Classification Metrics - Baseline Logistic Regression')
    tables += 1
    st.dataframe(score.loc[['0','1','accuracy'],:].sort_values(['split']))

    st.markdown('## __SMOTE__')
    st.markdown('One way we can handle imbalanced data sets is [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) or __S__ynthetic __M__inority __O__versampling __Te__chnique.')
    st.write('SMOTE is an over-sampling technique that creates  synthetic samples of the minority class through KNN. SMOTE randomly selects one of the minority class and then finds n of it''s K-Nearest Neighbors, from these neighbors, one is selected at random and connects a line through the feature-space. The new sample is then generated by selecting a point between the 2 neighbors in the feature space. This is repeated until the over-sampling % data set is acheived.')

    st.code('''lr_smote = LogisticRegression()
over = SMOTE()

model = make_pipeline(over,lr_smote)
model.fit(X_train.values,y_train.values.ravel())''',language='pythone')

    lr_smote = LogisticRegression()
    over = SMOTE()
    model = make_pipeline(over,lr_smote)
    model.fit(X_train.values,y_train.values.ravel())

    st.write('Again, we are able to get a high AUC as seen in Figure 3, however looking at the classification metrics, we notice that we were able to boost the recall for the negative class from 0.41 to 0.87. We were able to classify 47 of the 54 negative records in the test set correctly. However, our precision fell as we are incorrectly classifying more positive examples as negative. The use-case of the problem would help determine which is more important. For example, if we wanted to always classify abalone with fewer than 5 rings correct most of the time because they were worth vastly more than abalone with greater than 5 rings, this tuning would be appropriate.')
    score,figures = get_classification_metrics(model,X_train.values,y_train.values.ravel(),X_test.values,y_test.values.ravel(),model_name='LR SMOTE',figures=figures)
    scores.append(score)
    st.markdown(f'### Table {tables}: Classification Metrics - Logistic Regression with SMOTE')
    tables += 1
    st.dataframe(score.loc[['0','1','accuracy'],:].sort_values(['split']))    

    st.markdown('## __Under Sampling and SMOTE__')
    st.write('Another way to deal with imbalanced data set is to under-sample the majority class. We can do this with a `RandomUnderSampler()`. We can use this in-conjunction with SMOTE, doing this allows us to balance the data set without creating too many synthetic examples while also not ignoring too many records of actual data from the majority class.')
    st.code('''lr_smote = LogisticRegression()
over = SMOTE(sampling_strategy=0.20)
under = RandomUnderSampler(sampling_strategy=0.20)
model = make_pipeline(over,under,lr_smote)
model.fit(X_train.values,y_train.values.ravel())''',language='python')

    st.write('Looking at `Table 9`, we can see that using both SMOTE and Under-Sampling allowed us to raise the f1-score from 0.48 to 0.64. We lost a little recall for the negative class, however the precision increased as we are predicting less false negatives.')

    lr_smote = LogisticRegression()
    over = SMOTE(sampling_strategy=0.20)
    under = RandomUnderSampler(sampling_strategy=0.20)
    model = make_pipeline(over,under,lr_smote)
    model.fit(X_train.values,y_train.values.ravel())
    score,figures = get_classification_metrics(model,X_train.values,y_train.values.ravel(),X_test.values,y_test.values.ravel(),model_name='LR SMOTE w/ Under Sampling',figures=figures)
    scores.append(score)
    st.markdown(f'### Table {tables}: Classification Metrics - Logistic Regression with SMOTE and Under-Sampling')
    tables += 1
    st.dataframe(score.loc[['0','1','accuracy'],:].sort_values(['split']))

    st.markdown('## __Adding in PCA__')
    st.write('In our initial EDA, one of the observations about the weight features was that they are highly correlated. This multicollinearity of the features can cause issues in modeling. One simple explaination of how multicollinearity can affect a model is in the betas of a linear model. If two features are highly correlated, the betas of those features may not make numerical sense. For example, Test Grades and Report Card Grades of high school students to predict IQ, the beta for the feature `Report Card Grade` may be negative, but intuitively this does not make sense and there should be a positive relationship with IQ.')
    st.write('One way to deal with multicollinearity is __P__rincipal __C__omponents __A__nalysis or PCA. Some other example methods of feature reduction include: Feature Selection (recursive or univariate), and AutoEncoders.')
    st.write('PCA works by calculating EigenVectors and EigenValues for the independent features. The goal of PCA is to find the Principal Components that explain a large portion of the variance in the data. Doing this allows us to potentially convert a 9-dimensional data set into a 3-dimensional data set to remove the highly correlated dimensions. One of the draw-backs of PCA is that we lose some explainability of the feature set. So if the goal of the modeling exercise is to have a simple and explainable model, then maybe feature elimination would be a better route.')

    st.code('''lr_smote = LogisticRegression()
pca = PCA(n_components=0.95)
over = SMOTE(sampling_strategy=0.20)
under = RandomUnderSampler(sampling_strategy=0.20)

model = make_pipeline(over,under,pca,lr_smote)
model.fit(X_train.values,y_train.values.ravel())''',language='python')

    lr_smote = LogisticRegression()
    pca = PCA(n_components=0.95)
    over = SMOTE(sampling_strategy=0.20)
    under = RandomUnderSampler(sampling_strategy=0.20)

    model = make_pipeline(over,under,pca,lr_smote)
    model.fit(X_train.values,y_train.values.ravel())
    score,figures = get_classification_metrics(model,X_train.values,y_train.values.ravel(),X_test.values,y_test.values.ravel(),model_name='LR SMOTE w/ Under Sampling and PCA',figures=figures)
    scores.append(score)
    st.markdown(f'### Table {tables}: Classification Metrics - Logistic Regression with SMOTE, Under-Sampling, and PCA')
    tables += 1
    st.dataframe(score.loc[['0','1','accuracy'],:].sort_values(['split']))

    st.markdown('## Pipeline Evaluation')
    st.markdown('Looking at the test classification results, we can see that the model pipeline with SMOTE and Under-Sampling has the highest recall and f1-score. Adding in PCA to the pipeline obtains similar results, we can do cross-fold validation to see if these pipelines truly have similar results, however according [Occams Razor](https://en.wikipedia.org/wiki/Occam%27s_razor), the simplier model is preferred and we can obtain an accuracy of 0.96 while simultaneously achieving a recall of 0.80 on the minority class by only using SMOTE and Under-Sampling.')
    scores = pd.concat(scores)
    st.markdown(f'### Table {tables}: Classification Metrics - Test Data Set')
    tables += 1
    st.dataframe(scores[scores.split=='test'].loc[['0','accuracy'],:].sort_values(['f1-score'],ascending=False).sort_index())

    st.markdown('## __AutoML - TPOT__')
    st.markdown("[TPOT](http://epistasislab.github.io/tpot/) is an AutoML Python library that automates the Pipline process using genetic algorithms. TPOT is a tool that iterates through different classifiers and hyperperameters allowing you to develop and iterate on model pipelines faster. Because TPOT does model selection and hyperperameter tuning, a single run of TPOT can take longer than fitting a single model. Because of this, I'll post images of how that works, but will not use the code directly in this app as it will take a long time to finish.")

    st.code('''
over = SMOTE(sampling_strategy=0.50)
under = RandomUnderSampler(sampling_strategy=0.75)
X_train_sample, y_train_sample = over.fit_resample(X_train,y_train)
X_train_sample, y_train_sample = under.fit_resample(X_train_sample, y_train_sample)

from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(generations=20, population_size=20, verbosity=2,)
pipeline_optimizer.fit(X_train_sample.values, y_train_sample.values.ravel())

pipeline_optimizer.export('models/tpot-clf.py')''',language='python')

    st.image(Image.open('img/tpot-training.png'))
    st.write('After 20 iterations, TPOT arrived at an optimal classifier using `GradientBoostingClassifier()` with the following hyperperameters')
    st.markdown('''* GradientBoostingClassifier()
    * learning_rate: 0.5
    * max_depth: 10
    * max_features: 0.25
    * max_features: 0.25 
    * min_samples_leaf: 3 
    * min_samples_split: 2
    * n_estimators: 100 
    * subsample: 0.90
    ''')

    st.markdown('With the TPOT suggested model and tuning, we can take these parameters and fit our final model and plot our classification metrics and ROC Curve.')
    st.code('''exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.25, min_samples_leaf=3, min_samples_split=2, n_estimators=100, subsample=0.9000000000000001)
exported_pipeline.fit(X_train_sample, y_train_sample)''',language='Python')
    over = SMOTE(sampling_strategy=0.50)
    under = RandomUnderSampler(sampling_strategy=0.75)
    X_train_sample, y_train_sample = over.fit_resample(X_train,y_train)
    X_train_sample, y_train_sample = under.fit_resample(X_train_sample, y_train_sample)
    exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.25, min_samples_leaf=3, min_samples_split=2, n_estimators=100, subsample=0.9000000000000001)
    exported_pipeline.fit(X_train_sample, y_train_sample)
    score,figures = get_classification_metrics(
        exported_pipeline,
        X_train_sample.values,y_train_sample.values.ravel(),
        X_test.values,y_test.values.ravel(),
        model_name=f'TPOT',
        figures=figures,
    )
    scores.append(score)
    st.markdown(f'### Table {tables}: Classification Metrics - TPOT with SMOTE and Under-Sampling')
    tables += 1
    st.dataframe(score.loc[['0','1','accuracy'],:].sort_values(['split']))
    st.write('Looking at `Table 12`, we can see that the f1-score and recall are higher than the logistic regression baseline model. However, the tree based model is not as accurate as the `LogisticRegression()` with SMOTE and Under-Sampling. Running TPOT for longer can help achieve a better performing model at the cost of time.')


@st.cache
def get_data():
    df = pd.read_csv('data/abalone2.csv')
    df[['infant','male']] = pd.get_dummies(df.sex,drop_first=True)
    df.loc[df.height==0.,'height'] = np.nan
    return df


def get_classification_metrics(model,X_train,y_train,X_test,y_test,model_name='default',figures=1):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:,1]
    test_prob = model.predict_proba(X_test)[:,1]

    train_scores = classification_report(y_train,train_pred,output_dict=True)
    test_scores = classification_report(y_test,test_pred,output_dict=True)

    train_scores = pd.DataFrame(train_scores).T
    test_scores = pd.DataFrame(test_scores).T
    train_scores['model'] = model_name
    test_scores['model'] = model_name
    train_scores['split'] = 'train'
    test_scores['split'] = 'test'

    train_roc = roc_curve(y_train,train_prob,drop_intermediate=False)
    test_roc = roc_curve(y_test,test_prob,drop_intermediate=False)

    train_auc = auc(train_roc[0],train_roc[1])
    test_auc = auc(test_roc[0],test_roc[1])

    lw=2
    fig, ax = plt.subplots()
    ax.plot(train_roc[0],train_roc[1],label=f'Train - AUC: {train_auc:.2f}',lw=lw)
    ax.plot(test_roc[0],test_roc[1],label=f'Test - AUC: {test_auc:.2f}',lw=lw)
    ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),label='No Skill',c='black',linestyle='--',lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Figure {figures}: Reciever Operator Characteristic Curve\n{model_name}')
    plt.plot()
    plt.legend()
    figures += 1
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test,test_pred),
        annot=True,
        cmap='Blues',
        fmt='d',
        ax=ax
    )
    plt.title(f'Figure {figures}: Confusion Matrix - Test\n{model_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.grid(False)
    st.pyplot(fig)
    figures += 1

    return pd.concat([train_scores,test_scores]),figures
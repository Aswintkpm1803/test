# -*- coding: utf-8 -*-
"""predictive analysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Jzq5LsGaIZulWyEXSgU2lYJvpw4QyQHs

# Best Fit Line
"""

import pandas as pd

"""**Importing Dataset**"""

df=pd.read_csv('/content/survey lung cancer.csv')
df.head()

"""**Primary Analysis**"""

df.shape

df.isnull().sum()

df.duplicated().sum()

"""**Defining Functions for Removing duplicates, Outliers, and encoding**"""

num_cols=df.select_dtypes(include='number')
cat_cols=df.select_dtypes(include='category')

def remove_duplicates(df):
  df=df.drop_duplicates()
  return df

def outliers_remove(df):
  for cols in num_cols:
    Q1=df[cols].quantile(0.25)
    Q3=df[cols].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    df=df[(df[cols]>=lower_bound) & (df[cols]<=upper_bound)]
  return df


feature_cols=['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ',
             'ALLERGY ','WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN',]

def encoding_feature_cols(df):
  for col in feature_cols:
    df[col]=df[col].map({1:0,2:1})
  return df


def gender_column_encoding(df):
  df['GENDER']=df['GENDER'].map({'M':1,'F':0})
  df['LUNG_CANCER']=df['LUNG_CANCER'].map({'YES':1,'NO':0})
  return df


df=remove_duplicates(df)
df=outliers_remove(df)
df=encoding_feature_cols(df)
df=gender_column_encoding(df)
df

"""**Splitting the data to train and test**

**Defining pipeline for scaling, feature selection and dimentionality reduction using PCA, applying polynomial features and Regression**

**Fitting the train data to the pipeline**
"""

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



x=df.drop(columns=['LUNG_CANCER'])
y=df['LUNG_CANCER']

# nullrem_transformed=FunctionTransformer(nullrem)
# outlieremover_transformed=FunctionTransformer(outlieremover)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a pipeline with polynomial features and logistic regression
pipe = Pipeline([

    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('poly', PolynomialFeatures(degree=6)),
    ('logistic', LogisticRegression())
])

# Fit the pipeline to the training data
pipe.fit(x_train, y_train)

y_pred=pipe.predict(x_test)
y_pred

np.array(y_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)

#display the confusion matrix using ConfusionMatrixDisplay function
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Yes','No']))
disp.plot()
plt.show()

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

precision_score(y_test, y_pred)

from sklearn.metrics import recall_score

recall_score(y_test, y_pred)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)
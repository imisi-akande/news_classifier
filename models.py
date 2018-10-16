import pandas as pd
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from vectorizer import tokenizer
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

df_all = pd.read_csv("csv_data/Clean_Overall_DataSet.csv")  
   


X_body_text = df_all.review.values
y = df_all.sentiment.values 

# Apply TfidfVectorizer to obtain Term Frequency and Inverse Document Frequency

tfidfVectorize = TfidfVectorizer(ngram_range=(1,2),
                                max_df= 0.85, 
                                min_df= 0.01,
                                tokenizer=tokenizer,)

X_body_tfidf = tfidfVectorize.fit_transform(X_body_text)

# Split Datasets into 20% Test and 80% Train datasets

X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, 
test_size = 0.2, random_state=1234)

# Apply logistic regression Algorithm
print 'server is running (Training Logistic Regression model)'

logistic_regression_body = LogisticRegression(penalty='l1')

logistic_regression_body.fit(X_body_tfidf_train, y_body_train)

y_body_pred = logistic_regression_body.predict(X_body_tfidf_test)

dest = os.path.join('./pkl_objects')

print 'server is still running (End Training Logistic Regression model, next XGBoost model Training)'


# Apply XGBoost Algorithm

xgb_body = XGBClassifier()

xgb_body.fit(X_body_tfidf_train, y_body_train)

y_xgb_body_pred = xgb_body.predict(X_body_tfidf_test)

print 'server is still running (End XGBoost model Training)'





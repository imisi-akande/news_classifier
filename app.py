from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, SelectField, validators
from models import tfidfVectorize
from sklearn import linear_model
import pickle
import sqlite3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from models import logistic_regression_body
from models import xgb_body

app = Flask(__name__)

import warnings
warnings.filterwarnings("ignore")

###### Preparing the Classifier
cur_dir = os.path.dirname(__file__)

clf = linear_model.SGDClassifier()                  
print 'server is still running (almost ready)'
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0:'real', 1:'fake'}
    X = tfidfVectorize.transform([document])
    y = logistic_regression_body.predict(X)[0]
    proba = np.max(logistic_regression_body.predict_proba(X))
    return label[y], proba

def xgbclassify(document):
    label = {0:'real', 1:'fake'}
    X = tfidfVectorize.transform([document])
    y = xgb_body.predict(X)[0]
    proba = np.max(xgb_body.predict_proba(X))
    return label[y], proba

def train(document, y):
    labels = [0, 1]
    X =tfidfVectorize.transform([document])
    clf.partial_fit(X, [y], classes=np.unique(labels), sample_weight=None)

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('NOW'))", (document, y))
    conn.commit()
    conn.close()

app = Flask(__name__)


class ReviewForm(Form):
    newsreview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
    comp_select = SelectField(choices=[('Logistic Regression', 'Logistic Regression'),
                                           ('XGBoost', 'XGBoost')])

@app.route('/home')
def home():
    return render_template('home.html', title='Home')

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/')
def index():
    form =ReviewForm(request.form)
    return render_template('reviewform.html', form=form, title='Enter News')

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    algo = request.form['comp_select']
    review = request.form['newsreview']
    if request.method == 'POST' and form.validate():
        if algo == 'Logistic Regression':
            y, proba = classify(review)
        elif algo == 'XGBoost':
            y, proba = xgbclassify(review)           
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                algos=algo,
                                probability=round(proba*100, 2),
                                title='Result'
                               )
    return render_template('reviewform.html', form=form, title='Enter News')

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'real': 0, 'fake': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html', title='Completed')

if __name__ == '__main__':
    app.run(debug=True)


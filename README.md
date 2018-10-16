# News Review Sentiment Classifier
## News classifier built with Flask framework

This sample project is a post-phase of applying Machine learning algorithms for detection of fake news. The Machine learning process involves data scraping from three different sources(NYT, Guardian, Kaggle), data wrangling, data preparation, training  logistic regresion model and XGBoost algorithms on our data sets and evaluation. This project employ about 105747 data samples. The fakeness level takes categorical values of zeros and ones. 0 for real news 1 for fake news.

## Dependencies

To setup and run the sample code, you're going to install Python2.7, pip, virtualenvwrapper.

## Setup

To setup and run the sample code you need to clone the repo
        
1. Create a virtualenv to isolate our package dependencies locally
    virtual env
    
2. Activate the virtual environment
    source env/bin/activate
    
3. Install Python Requirements:

        pip install -r requirements.txt
        
4. For the server run :
        export FLASK_APP = app.py 
        flask run 
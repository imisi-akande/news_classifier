import pickle
import sqlite3
import numpy as np
import os
from models import tfidfVectorize
from models import logistic_regression_body
from sklearn import linear_model

"""TO UPDATE CLASSIFIER """

def update_model(db_path, model, batch_size=10000):
    clf = linear_model.SGDClassifier()   	
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT * from review_db')

	results = c.fetchmany(batch_size)
	while results:
		data = np.array(results)
		X = data[:, 0]
		y = data[:, 1].astype(int)

		classes = np.array([0, 1])
		X_train = tfidfVectorize.transform(X)
		clf.partial_fit(X_train, y, classes=classes)
		results = c.fetchmany(batch_size)

	conn.close()
	return None

cur_dir = os.path.dirname(__file__)

clf2 = pickle.load(open(os.path.join(cur_dir,
					'pkl_objects',
					'classifier.pkl'), 'rb'))

db = os.path.join(cur_dir, 'reviews.sqlite')

update_model(db_path=db, model=clf2, batch_size=10000)

pickle.dump(clf2, open(os.path.join(cur_dir,
						'pkl_objects', 'classifier.pkl'), 'wb'))

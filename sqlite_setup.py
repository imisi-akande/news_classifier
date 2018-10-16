import sqlite3
import os

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db'\
          ' (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'Students expressed their fear over a Trump presidency in messages to each other that were being shared on Twitter today'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('NOW'))", (example1, 1))

example2 = 'The parliamentarian who was elbowed by Justin Trudeau said she has been left fending off personal attacks, including accusations that she is “crying wolf”, in the wake of the high-profile incident.'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('NOW'))", (example2, 0))
conn.commit()
conn.close()



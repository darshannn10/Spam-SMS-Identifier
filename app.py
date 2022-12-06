import re
import string
from flask import Flask, render_template, request
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os
import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import NaiveBayesClassifier
import pickle

print('The scikit-learn version is {}.'.format(sklearn.__version__))

app = Flask(__name__)


# Defining a function to remove Punctuations!
string.punctuation

def rm_punc(text):
    new_text = ''.join(
        [char for char in text if char not in string.punctuation])
    return new_text


def tok(text):
    tokens = re.split('\W+', text)
    return tokens


stopwords = nltk.corpus.stopwords.words('english')


def rm_sw(text):
    clean_text = [word for word in text if word not in stopwords]
    return clean_text


ps = nltk.PorterStemmer()


def stemming(tokenized_text):
    stemmed_text = [ps.stem(word) for word in tokenized_text]
    return stemmed_text


def final_text(stemmed_text):
    get_final_text = " ".join([word for word in stemmed_text])
    return get_final_text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    vectorizer = pickle.load(open('./machine_learning/vectorizer.pkl', 'rb'))
    tfidf_transformer = pickle.load(
        open('./machine_learning/tfidf_transformer.pkl', 'rb'))
    svm = pickle.load(open('./machine_learning/svm.pkl', 'rb'))

    if request.method == 'POST':
        msg = request.form['message']

        msg = msg.lower()
        msg = rm_punc(msg)
        msg = tok(msg)
        msg = rm_sw(msg)
        msg = stemming(msg)
        msg = final_text(msg)
        msg = vectorizer.transform([msg])
        msg = tfidf_transformer.transform(msg)
        msg = svm.predict(msg)

    return render_template('index.html', prediction=msg, messsage=msg)


if __name__ == '__main__':
    app.run(debug=True)

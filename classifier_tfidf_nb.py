import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocess import Preprocess
import fasttext
from unidecode import unidecode
import os 
from collections import Counter
 
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix

def score(y_true, y_pred):
    print(f"Classification Report \n {classification_report(y_true, y_pred)}")
    

def lexicons():
    data = pd.read_csv('data/lexicons.csv', encoding='utf-8')
    positive = data['POSITIVE'].dropna()
    negative = data['NEGATIVE'].dropna()
    p_data = []
    n_data = []
    for p in positive:
        if(type(p) is str and p.strip()!=""):
            p_data.append((unidecode(p.lower().strip()),2))
    for n in negative:
        if(type(n) is str and n.strip()!=""):
            n_data.append((unidecode(n.lower().strip()),0))
    
    return p_data+n_data
def usefull_lexicons():
    data = pd.read_csv('data/usefull_lexicons.csv', encoding='utf-8')
    positive = data['POSITIVE'].dropna()
    negative = data['NEGATIVE'].dropna()
    p_data = []
    n_data = []
    for p in positive:
        if(type(p) is str and p.strip()!=""):
            p_data.append((unidecode(p.lower().strip()),2))
    for n in negative:
        if(type(n) is str and n.strip()!=""):
            n_data.append((unidecode(n.lower().strip()),0))
    
    return p_data+n_data


def load_posts():
    df = pd.read_csv('data/sentiment140/training.1600000.processed.noemoticon.csv', header=None, encoding='ISO-8859-1', names= ["target", "ids", "date", "flag", "user", "text"])
    print("sentiment140")
    print(df.head())
    sent_data = [(p, {0:0, 4:2}[t]) for p, t in df[['text', 'target']].to_numpy()]
    return sent_data

usefull_lexicons_ = usefull_lexicons()
lexicons_ = lexicons()
sentiment140_data = load_posts()

X, y = [p[0] for p in sentiment140_data],[p[1] for p in sentiment140_data]


X, y =  Preprocess().fit_transform(X, y)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=23)

X_train += [l[0] for l in usefull_lexicons_+lexicons_]
y_train += [l[1] for l in usefull_lexicons_+lexicons_]

print(f"Proportion of classes on train set: {Counter(y_train)}")
print(f"Proportion of classes on test set: {Counter(y_test)}")

tfidf =  TfidfVectorizer(ngram_range=(1, 2),lowercase=True, token_pattern=r'[^\s]+')

vectors = tfidf.fit_transform(X_train)

model = ComplementNB()
model.fit(vectors, y_train)

print(f"score on train: {model.score(vectors, y_train)}")
print(f"score on test:  {model.score( tfidf.transform(X_test), y_test)}")


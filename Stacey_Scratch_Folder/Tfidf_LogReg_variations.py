# import pandas as pd
# import numpy as np
# import re
#Importing everything from NLP Week 1 - following that as a guide for now
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import random
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
pd.options.display.max_rows = 100
pd.options.display.width = 150
RANDOM_SEED = 696

WikiLarge_Train_df = pd.read_csv(r'/Users/staceybruestle/Documents/Documents/Education/MADS/Coursework/aSIADS696- Milestone 2/Readability Project/Data/WikiLarge_Train.csv')#, \

train_df, dev_df, test_df = np.split(WikiLarge_Train_df.sample(frac=1, random_state= RANDOM_SEED), 
                       [int(.8*len(WikiLarge_Train_df)), int(.9*len(WikiLarge_Train_df))], axis = 0)
#Make list of labels
y_train = list(train_df.label)
y_dev = list(dev_df.label)
y_test = list(test_df.label)
# Shapes: 333414, 41677, 41677 all with 5 columns

# Create the Dummy Classifiers to use as reference for other scores
def dummyClassifierScores(train_df, dev_df, vectorizer, min_df=1, stop_words= None, strategy='uniform'):
    # Stratgeies 'uniform' and 'most_frequent'
    X_train = vectorizer.fit_transform(train_df.original_text)
    X_dev = vectorizer.transform(dev_df.original_text)

    dummy = DummyClassifier(strategy= strategy, random_state = RANDOM_SEED, constant=None)
    dummy.fit(X_train, y_train)

    # Generate the predictions
    dev_preds = dummy.predict(X_dev)

    # Score the predictions
    f1_dummy = f1_score(y_dev, dev_preds, average='macro')

    return f1_dummy

# Create a function for the steps so we can run it for various amounts of data to see the difference
# --  the function returns the macro-averaged F1 score on the dev data and the dummy if requested
def train_and_score(train_df, dev_df, min_df=1, max_df = 1.0, max_iter=100, C=1.0, \
                    ngram_range=(1,1), stop_words= None, dummy='no', strategy='uniform'):
    # Fit a new TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df= min_df, max_df= max_df, stop_words= None, ngram_range= ngram_range)
    X_train = vectorizer.fit_transform(train_df.original_text)

    #Get the labels
    y_train = list(train_df.label)

    #Fit the data to a Logistic Regression Classifier
    clf = LogisticRegression(random_state=RANDOM_SEED, max_iter = max_iter, C= C, multi_class='ovr', solver= 'newton-cholesky')
    clf.fit(X_train, y_train)

    # Generate the dev data
    X_dev = vectorizer.transform(dev_df.original_text)
    y_dev = list(dev_df.label)

    # Generate the predictions
    lr_tiny_dev_preds = clf.predict(X_dev)

    # Score the predictions
    f1 = f1_score(y_dev, lr_tiny_dev_preds, average='macro')

    if dummy== 'yes':
        f1 = (f1, dummyClassifierScores(train_df, dev_df, vectorizer, min_df, stop_words, strategy= strategy))

    return f1

# lr_score, dummy_score = train_and_score(train_df, dev_df, dummy='yes')
# print("All defaults with uniform dummy -\n  Logistic Regression score:", lr_score, "  Random Dummy score:", dummy_score )
# Log Reg Score: 0.6834110283504679
# Uniform (Random) Score: 0.49840355363816646

#Need to create the vectorizer to run dummyClassifierScores by itself
# This scored 0.333061289806369
# vectorizer = TfidfVectorizer()
# print("All defaults - most frequent dummy:", dummyClassifierScores(train_df, dev_df, vectorizer, strategy= 'most_frequent') )


print("\ndefault with stopwords:", train_and_score(train_df, dev_df, stop_words= 'english') )
#Score:

# train_and_score(train_df, dev_df, min_df=1, max_df = 1.0, max_iter=100, C=1.0, \
#                     ngram_range=(1,1), stop_words= None, dummy='no', strategy='uniform')



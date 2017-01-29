import os                                # accessing directory of files
import pandas as pd                      # storing the data
from bs4 import BeautifulSoup            # removing HTML tags
import re                                # text processing with regular expressions
from gensim.models import word2vec       # embedding algorithm
import numpy as np                       # arrays and other mathy structures     
from tqdm import tqdm                    # timing algorithms
from gensim import models                # doc2vec implementation
from random import shuffle               # for shuffling reviews
from sklearn.linear_model import LogisticRegression
import nltk.data                         # sentence splitting
from keras.models import Sequential      # deep learning (part 1)
from keras.layers import Dense, Dropout  # deep learning (part 2)


#####################
#   Load the Data   #
#####################

def load_data( directory_name ):
    # load dataset from directory to Pandas dataframe
    data = []
    files = [f for f in os.listdir('../../../aclImdb/' + directory_name)]
    for f in files:
        with open('../../../aclImdb/' + directory_name + f, "r", encoding = 'utf-8') as myfile:
            data.append(myfile.read())
    df = pd.DataFrame({'review': data, 'file': files})
    return df

# load training data
train_pos = load_data('train/pos/')
train_neg = load_data('train/neg/')
# load test data
test_pos = load_data('test/pos/')
test_neg = load_data('test/neg/')
# load unsupervised data
unsup = load_data('train/unsup/')


#######################
#   Process Reviews   #
#######################

def clean_str( string ):
    # Function that cleans text using regular expressions
    string = re.sub(r' +', ' ', string)
    string = re.sub(r'\.+', '.', string)
    string = re.sub(r'\.(?! )', '. ', string)    
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'m", " \'m", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"\:", " : ", string)
    string = re.sub(r'\"', ' " ', string)
    string = re.sub(r'\/', ' / ', string)
    return string

def review_to_wordstring( review ):
    #
    # Function to turn each review into a list of words
    #
    # 1. Remove HTML tags 
    review = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Process punctuation, excessive periods, missing spaces
    review = clean_str(review) 
    #
    # 3. remove white spaces
    review = review.strip()
    #
    # 4. return lowercase collection of words
    wordlist = review.lower().split()
    #
    # 5. Join the list of words back into one string, with words separated by space
    return ( " ".join( wordlist ) )

def corpus_to_list( df ):
    # Turns dataframe df of reviews into a list (each corresponding to a review) of lists of words
    return [ review_to_wordstring(df.ix[idx, 'review']) for idx in tqdm(df.index) ]

#####################
#   Get Embedding   #
#####################

def get_embedding():
    #
    # just return the reviews, yo!
    train_pos_sentences = corpus_to_list(train_pos)
    train_neg_sentences = corpus_to_list(train_neg)
    test_pos_sentences = corpus_to_list(test_pos)
    test_neg_sentences = corpus_to_list(test_neg)
    train_sentences = train_pos_sentences + train_neg_sentences
    test_sentences = test_pos_sentences + test_neg_sentences
    #
    return train_sentences, test_sentences

    
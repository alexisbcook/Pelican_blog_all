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
from classes import Doc2VecUtility
from classes import Word2VecUtility


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

def review_to_wordlist( review ):
    #
    # Function to turn each review into a list of words
    #
    # 1. Process punctuation, excessive periods, missing spaces 
    review = clean_str(review)
    #
    # 2. Remove HTML tags 
    review = BeautifulSoup(review, "lxml").get_text()
    #
    # 3. remove white spaces
    review = review.strip()
    #
    # 4. return lowercase collection of words
    wordlist = review.lower().split()
    #
    # 5. Return the list of words
    return wordlist

def makeReviewSequence( review, model, index2word_set, num_features ):
    #
    # Gets sequence of word2vec features for given review
    #
    # 1. get list of words in review
    words = review_to_wordlist(review)
    #
    # 2. Loop over each word in the review 
    #    If it is in the model's vocab, append its feature vector
    return np.array([model[w] for w in words if w in index2word_set])

def getReviewSequences( df, model, num_features ):
    #
    # Given a df of reviews, calculate sequence of word2vec features for each one 
    # 
    # 1. get list of words in model
    index2word_set = set(model.index2word)
    #
    # 2. Return array of RNN inputs for each review in df
    return np.array([makeReviewSequence(df.ix[idx, 'review'], model, index2word_set, num_features) for idx in tqdm(df.index)])

def get_embedding():
    #
    # 1. load the model
    if not os.path.isfile('models/w2v'):
        Word2VecUtility.get_embedding()
    model = models.Doc2Vec.load("models/w2v")
    #
    # 2. Obtain train data embeddings 
    num_features = 300
    pos_w2vRNN_train = getReviewSequences(train_pos, model, num_features)
    neg_w2vRNN_train = getReviewSequences(train_neg, model, num_features)
    w2vRNN_train = list(np.append(pos_w2vRNN_train, neg_w2vRNN_train, axis=0))
    #
    # 3. Obtain test data embeddings
    pos_w2vRNN_test = getReviewSequences(test_pos, model, num_features)
    neg_w2vRNN_test = getReviewSequences(test_neg, model, num_features)
    w2vRNN_test = list(np.append(pos_w2vRNN_test, neg_w2vRNN_test, axis=0))

    return [w2vRNN_train, w2vRNN_test]

    
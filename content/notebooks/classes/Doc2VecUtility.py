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


##########################
#   Doc2Vec Parameters   #
##########################

num_features = 300    # Word vector dimensionality                      
min_word_count = 1    # Minimum word count                        
num_workers = 8       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-4   # Downsample setting for frequent words


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

class LabeledLineReview( object ):
    def __init__(self, dflist):
        self.dflist = dflist

    def __iter__(self):
        for df in self.dflist:
            for idx in tqdm(df.index):
                yield models.doc2vec.LabeledSentence(review_to_wordlist(df.ix[idx, 'review']), [df.ix[idx, 'file']])

    def to_array(self):
        self.reviews = []
        for df in self.dflist:
            for idx in tqdm(df.index):
                self.reviews.append(models.doc2vec.LabeledSentence(review_to_wordlist(df.ix[idx, 'review']), [df.ix[idx, 'file']]))
        return self.reviews

    def reviews_perm(self):
        shuffle(self.reviews)
        return self.reviews


#####################
#   Train Doc2Vec   #
#####################

def train():
    #
    # Trains the doc2vec model
    #
    # 1. Get all reviews together
    reviews = LabeledLineReview([train_pos, train_neg, test_pos, test_neg, unsup])
    #
    # 2. Define the model
    model = models.Doc2Vec(workers = num_workers, \
                           size = num_features, min_count = min_word_count, \
                           window = context, sample = downsampling, negative = 5)
    model.build_vocab(reviews.to_array())
    #
    # 3. Train the model
    for epoch in tqdm(range(10)):
        model.train(reviews.reviews_perm())
    #
    # 4. Save the model
    model.init_sims(replace=True)
    model.save("models/d2v")


#####################
#   Get Embedding   #
#####################
            
def get_embedding():              
    #
    # Returns embedding and labels, training model if necessary
    #
    # 1. Load the saved model.
    #   (If the model is not already saved, train the model)
    if not os.path.isfile('models/d2v'):
        train()
    model = models.Doc2Vec.load("models/d2v")
    #
    # 2. Obtain train data embeddings and labels
    train_array = np.zeros((25000, num_features))
    train_tags = list(train_pos['file'].values) + list(train_neg['file'].values)
    for idx , val in enumerate(train_tags):
        train_array[idx] = model.docvecs[val]
    #
    # 3. Obtain test data embeddings and labels
    test_array = np.zeros((25000, num_features))
    test_tags = list(test_pos['file'].values) + list(test_neg['file'].values)
    for idx , val in enumerate(test_tags):
        test_array[idx] = model.docvecs[val]
    #
    # 4. Return embeddings and labels
    return train_array, test_array
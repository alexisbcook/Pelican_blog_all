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


###########################
#   Word2Vec Parameters   #
###########################

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


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

def getAvgFeatureVecs( df, model, num_features ):
    #
    # Given a df of reviews, calculate the average feature vector for each one 
    # 
    index2word_set = set(model.index2word)
    reviewFeatureVecs = []
    for idx in tqdm(df.index):
        to_append = makeAvgFeatureVec(df.ix[idx, 'review'], model, index2word_set, num_features)
        reviewFeatureVecs.append(to_append)
    reviewFeatureVecs = np.array(reviewFeatureVecs)
    return reviewFeatureVecs

def makeAvgFeatureVec( review, model, index2word_set, num_features ):
    #
    # Averages all of the word vectors in a given review
    #
    # 1. Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    # 2. Initialize number of words in review
    nwords = 0.
    # 
    # 3. Loop over each word in the review 
    #    If it is in the model's vocab, add its feature vector to the total
    words = review_to_wordlist(review)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # 
    # 4. Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    # 5. Return the average word vector
    return featureVec

def review_to_wordlist( review, only_words = False ):
    #
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Process punctuation, excessive periods, missing spaces
    review = clean_str(review)
    #
    # 3. (Optionally) remove non-letters / non-words
    if only_words:
        review = re.sub("[^a-zA-Z]"," ", review)
    #
    # 4. Convert words to lower case and split into list
    words = review.lower().split()
    #
    # 5. Return a list of words
    return(words)

def review_to_lists_of_lists( review, only_words = False ):
    # 
    # Function to turn each review into a list of sentences, 
    # where each sentence is a list of words
    #
    # 1. Process punctuation, excessive periods, missing spaces 
    review = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove HTML tags 
    review = clean_str(review)
    #
    # 3. (Optionally) remove non-letters / non-words
    if only_words:
        review = re.sub("[^a-zA-Z]"," ", review)
    #
    # 4. Use the NLTK tokenizer to split the review into list of sentences
    #   (getting rid of extra spaces at front/back)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 5. Loop over each sentence to get list of list of lowercase words
    sentences = []
    for raw_sentence in raw_sentences:
        # Convert to lowercase and split into list of words
        raw_sentence = raw_sentence.lower().split()
        # If a sentence is not long enough, skip it
        if len(raw_sentence) > 1:
            # add list of words to returned object
            sentences.append( raw_sentence )
    #
    # 6. Return the list of sentences (each sentence is a list of words,
    # so returns a list of lists)
    return sentences

def corpus_to_list( df, tokenizer ):
    # Turns dataframe of reviews into a list of sentences,
    # where each sentence is a list of words
    # and sentences are derived from *all reviews* in dataframe df
    from tqdm import tqdm
    sentences = []
    for idx in tqdm(df.index):
        to_append = review_to_lists_of_lists(df.ix[idx, 'review'])
        sentences += to_append
    return sentences

######################
#   Train Word2Vec   #
######################

def train():
    #
    # Trains the word2vec model
    #
    # 1. Assemble all reviews
    train_pos_sentences = corpus_to_list(train_pos, tokenizer)
    train_neg_sentences = corpus_to_list(train_neg, tokenizer)
    test_pos_sentences = corpus_to_list(test_pos, tokenizer)
    test_neg_sentences = corpus_to_list(test_neg, tokenizer)
    unsup_sentences = corpus_to_list(unsup, tokenizer)
    sentences = train_pos_sentences + train_neg_sentences + test_pos_sentences + test_neg_sentences + unsup_sentences
    #
    # 2. Initialize and train the model 
    model = word2vec.Word2Vec(sentences, workers = num_workers, \
                size = num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    #
    # 3. Save the model
    model.init_sims(replace=True)
    model.save("models/w2v")
        
def get_embedding():
    #
    # Returns embedding and labels, training model if necessary
    #
    # 1. Load the saved model.
    #   (If the model is not already saved, train the model)
    if not os.path.isfile('models/w2v'):
        train()
    model = models.Doc2Vec.load("models/w2v")
    #
    # 2. Obtain train data embeddings 
    pos_w2v_train = getAvgFeatureVecs(train_pos, model, num_features)
    neg_w2v_train = getAvgFeatureVecs(train_neg, model, num_features)
    w2v_train = np.append(pos_w2v_train, neg_w2v_train, axis=0)
    #
    # 3. Obtain test data embeddings
    pos_w2v_test = getAvgFeatureVecs(test_pos, model, num_features)
    neg_w2v_test = getAvgFeatureVecs(test_neg, model, num_features)
    w2v_test = np.append(pos_w2v_test, neg_w2v_test, axis=0)
    #
    # 4. Return all embeddings
    return w2v_train, w2v_test
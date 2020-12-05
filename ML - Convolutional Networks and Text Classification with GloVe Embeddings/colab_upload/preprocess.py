import multiprocessing

import gensim
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords


def preprocess_string(string):
    '''Tokenizes and removes stopwords'''

    # Convert string to list of lowercase tokens, which are not too long (1 character) or too short (>15 characters)
    tokenized_string = simple_preprocess(string
                                         , min_len = 3 # default 2
                                         , max_len = 15) # default 15

    no_stops_string = [word for word in tokenized_string if word not in stopwords.words('english')]

    return no_stops_string


def preprocess_texts(texts):

    pool=multiprocessing.Pool(multiprocessing.cpu_count())

    preprocessed_texts = pool.map(preprocess_string, texts)

    pool.close()

    return preprocessed_texts
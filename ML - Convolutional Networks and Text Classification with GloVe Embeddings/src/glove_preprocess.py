import multiprocessing
import numpy as np
import json
import os

import gensim
import gensim.downloader
glove_50 = gensim.downloader.load('glove-wiki-gigaword-50')
# nltk.download('stopwords')


def doc_to_vec(text):
    
    lst_vec = []
    
    for word in text:
        try:
            lst_vec.append(glove_50.get_vector(word))
        except:
            pass # Ignore word if it's not in vocabulary of glove_50
    
    vec = np.array(lst_vec)
    
    return np.concatenate((np.min(vec, axis=0), np.max(vec, axis=0), np.mean(vec, axis=0)))

# NON PARALLELIZED
def docs_to_vecs(texts):
    
    doc_vecs = []
    
    for text in texts:
        doc_vecs.append(doc_to_vec(text))
    
    return np.array(doc_vecs)


# PARALLELIZED, NOT SURE IF WORKS
# def docs_to_vecs(texts):
    
#     pool=multiprocessing.Pool(multiprocessing.cpu_count())

#     doc_vecs = pool.map(doc_to_vec, texts)

#     pool.close()
    
#     return np.array(doc_vecs)
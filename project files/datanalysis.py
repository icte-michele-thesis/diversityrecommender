#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:26:19 2017

@author: michele
"""

import json
import numpy as np
import pandas as pd
#import requests
#from bs4 import BeautifulSoup

#all_ratings = pd.read_csv("ml-20m/ratings.csv")
# all_ratings.head()
#all_movies = pd.read_csv("ml-20m/movies.csv")

def gettitles():
    all_links = pd.read_csv("ml-latest-small/links.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    movietitles = pd.merge(all_links, movies, on ='movieId', how='inner').drop(['movieId','tmdbId'],1)
    return movietitles.to_dict(orient = 'records')










# try some LSA with the final datasets
import random
import numpy as np
import string
import json
import time
import logging



# GENSIM
import gensim
from gensim import corpora, similarities
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


np.random.seed(42)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




def importMoviesfromJSON(JSONfile):
    data = [] # list of movies from json
    with open(JSONfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data

finaldata1 = importMoviesfromJSON('finaldata1')
finaldata2 = importMoviesfromJSON('finaldata2')
finaldata3 = importMoviesfromJSON('finaldata3')







    #==============================    6) PREPARE DICTIONARIES AND TFIDF MATRICES for finaldata1

    
features1 = [m['features'] for m in finaldata1]

# building the dictionary and the docterm matrix

f1_dictionary = corpora.Dictionary(features1)
f1_doc_term_matrix = [f1_dictionary.doc2bow(doc) for doc in features1]
                            
corpora.MmCorpus.serialize('LSA_finaldata1_CORPUS_full',f1_doc_term_matrix)
#LSA_IMDB_corpus = corpora.MmCorpus('LSA_IMDB_CORPUS')

# building the TFIDF matrix out of the docterm matrix
f1_tfidf = gensim.models.TfidfModel(f1_doc_term_matrix)
f1_tfidf_matrix = f1_tfidf[f1_doc_term_matrix]


    #==============================     7.1) APPLY LSA ON finaldata1
# prepare the model
f1_lsi = gensim.models.LsiModel(f1_tfidf_matrix, id2word=f1_dictionary, num_topics=40)

# train the model on all plots
f1_corpus_lsi = f1_lsi[f1_tfidf_matrix]

#LSA_IMDB_lsi.print_topics(50)

# index the plot similarities
f1_index = similarities.MatrixSimilarity(f1_corpus_lsi)#LSA_IMDB_lsi[LSA_IMDB_corpus_lsi])
f1_index.save('f1_INDEX_full.index')


    #==============================     10) PREPARE SIMILARITY QUERIES FOR GIVEN IMDBIDS


def get_test_movie(imdbid_test,finaldata):
    """ it returns the bag of words corresponging to the plot of a given movie """
    testfeatures = [f for f in finaldata if f['imdbid'] == imdbid_test]
    features = testfeatures[0]['features']
    features_bow = f1_dictionary.doc2bow(features) # dictionary is can be a parameter to this function
    return features_bow # test_IMDB_bow,test_WIKI_bow
    


# test on IMDB LSA
def get_IMDB_LSA_similaritylist(imdbid):
    start = time.time()
    test_feature_bow = get_test_movie(imdbid,finaldata1)
    tfidf_bow = f1_tfidf[test_feature_bow] #  tfidf can be a parameter
    vec_lsi = f1_lsi[tfidf_bow] #             lsi can be a parameter
    sims_imdb_lsa = f1_index[vec_lsi]
    sims_imdb_lsa = sorted(enumerate(sims_imdb_lsa), key=lambda item: -item[1])
    end = time.time()
    print("Time elapsed: "+ str(end - start))
    return sims_imdb_lsa


# test imdbids:
#   245429 spirited away
#   133093 matrix
  
def getsimilartitles(imdbid,topn):    
    similarto = get_IMDB_LSA_similaritylist(imdbid)[0:topn]
    indexes = [f[0] for f in similarto]
    fin1 = np.array(finaldata1.copy())
    imdbidsofsims = [f['imdbid'] for f in list(fin1[indexes])]
    similarmovies = []
    movietitles = gettitles()
    for imdbid in imdbidsofsims:
        similarmovies.append([d['title'] for d in movietitles if d['imdbId']==imdbid][0])
    return similarmovies




topn = 10
get_IMDB_LSA_similaritylist(347149)[0:topn]

similars = getsimilartitles(11237,30)

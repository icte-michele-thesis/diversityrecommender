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

all_links = pd.read_csv("ml-latest-small/links.csv")



















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
f1_dictionary
f1_doc_term_matrix
f1_dictionary = corpora.Dictionary(features1)
f1_doc_term_matrix = [f1_dictionary.doc2bow(doc) for doc in features1]
                            
corpora.MmCorpus.serialize('LSA_finaldata1_CORPUS_full',f1_doc_term_matrix)
#LSA_IMDB_corpus = corpora.MmCorpus('LSA_IMDB_CORPUS')

# building the TFIDF matrix out of the docterm matrix
f1_tfidf = gensim.models.TfidfModel(f1_doc_term_matrix)
f1_tfidf_matrix = LSA_f1_tfidf[f1_doc_term_matrix]



    #==============================     7.1) APPLY LSA ON finaldata1

# prepare the model
f1_lsi = gensim.models.LsiModel(LSA_IMDB_tfidf_matrix, id2word=LSA_IMDB_dictionary, num_topics=40)

# train the model on all plots
f1_corpus_lsi = f1_lsi[f1_tfidf_matrix]

#LSA_IMDB_lsi.print_topics(50)

# index the plot similarities
f1_index = similarities.MatrixSimilarity(f1_corpus_lsi)#LSA_IMDB_lsi[LSA_IMDB_corpus_lsi])
f1_index.save('f1_INDEX_full.index')











    #==============================     10) PREPARE SIMILARITY QUERIES FOR GIVEN IMDBIDS


def get_test_movie(imdbid_test,imdbflag):
    """ it returns the bag of words corresponging to the plot of a given movie """
    testplot = [f for f in finaldata if f['_IMDb_ID'] == imdbid_test]
    if(imdbflag): # if True, it is an OMDB plot and the OMDB plot dictionary is used
        plot = testplot[0]['imdb_plot']
        plot_bow = LSA_IMDB_dictionary.doc2bow(tokenize_only(plot))
    else:
        plot = testplot[0]['plot']
        plot_bow = LSA_WIKI_dictionary.doc2bow(tokenize_only(plot))
    return plot_bow # test_IMDB_bow,test_WIKI_bow
    


# test on IMDB LSA
def get_IMDB_LSA_similaritylist(imdbid):
    start = time.time()
    test_IMDB_bow = get_test_movie(imdbid,True)
    tfidf_bow = LSA_IMDB_tfidf[test_IMDB_bow]
    vec_lsi = LSA_IMDB_lsi[tfidf_bow]
    sims_imdb_lsa = LSA_IMDB_index[vec_lsi]
    sims_imdb_lsa = sorted(enumerate(sims_imdb_lsa), key=lambda item: -item[1])
    end = time.time()
    print("Time elapsed: "+ str(end - start))
    return sims_imdb_lsa










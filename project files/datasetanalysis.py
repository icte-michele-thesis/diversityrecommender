#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:24:56 2017

METADATA SCRAPER FROM IMDB

@author: michele
"""
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing


import matplotlib.pyplot as plt
#import requests
#from bs4 import BeautifulSoup

#all_ratings = pd.read_csv("ml-20m/ratings.csv")
# all_ratings.head()
#all_movies = pd.read_csv("ml-20m/movies.csv")

links = pd.read_csv("ml-latest-small/links.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")



        
# import movie metadata from JSON FILE
metadata = []
with open('json_metadataFULL_final', encoding='utf-8') as datafile:
    metadata = json.load(datafile)        
metadatadf = pd.DataFrame(metadata).fillna(0)        


        
DE = [m['imdbid'] for m in metadata if 'Hayao Miyazaki' in m['director']]    
len(DE)    


imdbratings = pd.merge(ratings, links, on='movieId', how='inner')
# create ratings matrix        
Rdf = imdbratings.pivot(index = 'userId',columns = 'imdbId',values = 'rating').fillna(0)
R = Rdf.as_matrix()



Rnan = R
Rnan[Rnan == 0] = np.nan
means = np.nanmean(Rnan[:, 1:], axis=1)
stds = np.nanstd(Rnan[:, 1:], axis=1)

#user_ratings_mean = np.ma.masked_equal(R, 0).mean(axis=1) #np.mean(R, axis = 1)
#user_ratings_std = np.ma.masked_equal(R, 0).std(axis=1) #np.mean(R, axis = 1)


# inspect the ratings and rating means and standard deviations
# plt.hist(ratings.rating)
# plt.hist(user_ratings_mean)
# plt.hist(user_ratings_std)

# normalize R by demeaning and dividing by std
#R_zscored = (R - user_ratings_mean.reshape(-1, 1))/user_ratings_std.reshape(-1, 1)
Rnan_zscored = (Rnan - means.reshape(-1, 1))/stds.reshape(-1, 1)
Rnan_zscored[np.isnan(Rnan_zscored)] = 0      
# np.count_nonzero(Rnan_zscored[1])




# NOW WE HAVE THE MATRIX NORMALIZED!!!!!!! what to do???

# check if ratings match the 0 mean by extracting those above 0 for a given user
row1 = Rnan_zscored[120]
whr = np.where(row1>0.8)
R[0][whr]


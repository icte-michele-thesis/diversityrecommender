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
import re

import matplotlib.pyplot as plt
#import requests
#from bs4 import BeautifulSoup

#all_ratings = pd.read_csv("ml-20m/ratings.csv")
# all_ratings.head()
#all_movies = pd.read_csv("ml-20m/movies.csv")

links = pd.read_csv("ml-latest-small/links.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")



# MOVIE AND USER STATISTICS: MEAN AND STD FOR EACH USERID AND MOVIEID
userstats = ratings.groupby('userId', as_index=False).agg({'rating':[np.size,np.mean,np.std]}) # statistics on users
ratingswmstd = pd.merge(ratings, userstats, on ='userId', how='inner')
# get normalized ratings!!
ratingswmstd['normrating'] = (ratingswmstd['rating'] - ratingswmstd[('rating', 'mean')])/ratingswmstd[('rating', 'std')]
# leave only the normalized rating, userid and movieid in the new dataset
ratingswmstd[ratingswmstd['normrating']>0].movieId
# 6734 movies!! :
len(list(set(ratingswmstd[ratingswmstd['normrating']>0].movieId)))

moviesabovethreshold = pd.DataFrame(ratingswmstd[ratingswmstd['normrating']>0].movieId).drop_duplicates()
# select imdbids corresponding to the movies there!
links = pd.merge(links, moviesabovethreshold, on = 'movieId', how = 'inner')
linksimdb = list(links.imdbId) # the list of imdbid corresponding to movies rated above the normalized rating threshold








movieratings = pd.merge(ratingswmstd, movies, on ='movieId', how='inner')

moviegroups = movieratings.groupby('title')
moviestats = moviegroups.agg({'rating':[np.size,np.mean,np.std]}) # statistics on movies
moviestats[moviestats['rating']['size'] >= 50].sort_values([('rating', 'mean')], ascending=False) 


      
# import movie metadata from JSON FILE
metadata = []
with open('json_metadataFULL_final', encoding='utf-8') as datafile:
    metadata = json.load(datafile)        
metadatadf = pd.DataFrame(metadata).fillna(0)        



#==========================================================================================
#                            PREPROCESSING METADATA DATASET
#==========================================================================================


#============================1-FIND ALL N/A-===============================================
# ['company', 'director', 'keywords', 'genres', 'countries',
# 'imdbid', 'writer', 'languages', 'cast', 'release', 'original music']
numberofNA = {}
numberofNA['nancompany'] = len([m['company'] for m in metadata if m['company'] == "N/A"])
numberofNA['nandirector'] = len([m['director'] for m in metadata if m['director'] == "N/A"])
numberofNA['nankeywords'] = len([m['keywords'] for m in metadata if m['keywords'] == "N/A"])
numberofNA['nangenres'] = len([m['genres'] for m in metadata if m['genres'] == "N/A"])
numberofNA['nancountries'] = len([m['countries'] for m in metadata if m['countries'] == "N/A"])
numberofNA['nanwriter'] = len([m['writer'] for m in metadata if m['writer'] == "N/A"])
numberofNA['nanlanguages'] = len([m['languages'] for m in metadata if m['languages'] == "N/A"])
numberofNA['nancast'] = len([m['cast'] for m in metadata if m['cast'] == "N/A"])
numberofNA['nanrelease'] = len([m['release'] for m in metadata if m['release'] == "N/A"])
numberofNA['nanoriginal music'] = len([m['original music'] for m in metadata if m['original music'] == "N/A"])

origlinks = pd.read_csv("ml-latest-small/links.csv")
movietitles = pd.merge(movies, origlinks, on='movieId', how='inner').drop(['movieId','tmdbId'],1)

# extract titles for movies without certain features:
feature = 'director'
nodict = {}
#nodict['imdbId'] = [m['imdbid'] for m in metadata if m[feature] == "N/A"]
# use the one below if metadata is already vectorized, else use the above
nodict['imdbId'] = [m['imdbid'][0] for m in metadata if "N/A" in m[feature]]
nodf = pd.DataFrame.from_dict(nodict)
moviewo = pd.merge(movietitles, nodf, on='imdbId', how='inner')
# inspect the movies without feature
moviewo.head()




#============================2-TRANSFORM RELEASE DATE
#============================+-ADD MOVIELENS RELEASE DATE WHERE IS N/A
# change release date to only year!!!!!!
origlinks = pd.read_csv("ml-latest-small/links.csv")
movietitles = pd.merge(movies, origlinks, on='movieId', how='inner').drop(['movieId','tmdbId'],1)
    
# extract titles for movies without company:
feature = 'release'
nodict = {}
nodict['imdbId'] = [m['imdbid'] for m in metadata if m[feature] == "N/A"]
nodf = pd.DataFrame.from_dict(nodict)
moviewo = pd.merge(movietitles, nodf, on='imdbId', how='inner')
moviewo = moviewo.to_dict(orient = 'records')
# extract release date from movielens
for j,mm in enumerate(moviewo):
    txt = re.search(r'\d{4}', mm['title'])
    moviewo[j]['title'] = txt.group(0)

for i,m in enumerate(metadata):
    release = m['release']
    currimdb = m['imdbid']
    if(release != 'N/A'):
        txt = re.search(r'\d{4}', release)
        metadata[i]['release'] = txt.group(0)
    if(release == 'N/A'):
        metadata[i]['release'] = [h['title'] for h in moviewo if h['imdbId'] == currimdb][0]
        #print(metadata[i]['release'])




#============================3-VECTORIZE ALL METADATA-=====================================
# VECTORIZE METADATA
# because we need to concatenate them after cleaning, to make a single movie document
for i,m in enumerate(metadata):
    curr = {}
    curr['imdbid'] = m['imdbid']
    for key, value in m.items():
        if(type(value) != list and key != 'imdbid'):
            metadata[i][key] = [value]

# save to JSON!
with open('METADATAVECTORIZED.json', 'w') as fout:
    json.dump(metadata, fout)



#============================4-CLEAN UP TO 100 KEYWORDS-=====================================


# from 1 to 10 
#castnr = []
#for m in metadata:
#    castnr.append(len(m['cast']))
#plt.hist(castnr) # distribution of cast, no reason to limit this.   

keynr = []
for m in metadata:
    keynr.append(len(m['keywords']))
plt.hist(keynr) # distribution of keywords, we need to limit them  
plt.xlabel('# of keywords')
plt.ylabel('# of movies')
    
# limit also the keywords?
# from 1 to 661, so we should limit keywords to 100
metadataklim = metadata
for i,m in enumerate(metadataklim):
    if(len(m['keywords']) > 100): # cut-off at 100
        metadataklim[i]['keywords'] = m['keywords'][0:100]

with open('METADATAVECTKEYWORD100.json', 'w') as fout:
    json.dump(metadataklim, fout)





# remove features with high percentage of N/A:
    # original music
    # writer









        
DE = [m['imdbid'] for m in metadata if 'Hayao Miyazaki' in m['director']]    
len(DE)    

# EXPLORATORY ANALYSIS:

#get movie distribution by users:
movieratings.title.plot.hist(bins=90)
movieratings.title.value_counts().sort_values(ascending = True)[:25]


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


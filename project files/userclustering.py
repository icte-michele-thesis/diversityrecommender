#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:13:00 2017

@author: michele
"""

import json
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import re

import random
import string
import time
import logging

import matplotlib.pyplot as plt
#import requests
#from bs4 import BeautifulSoup

#all_ratings = pd.read_csv("ml-20m/ratings.csv")
# all_ratings.head()
#all_movies = pd.read_csv("ml-20m/movies.csv")

links = pd.read_csv("ml-latest-small/links.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

def getdataset(jsonfile):
    data = [] # list of movies from json
    with open(jsonfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data


dataset1 = getdataset('finaldata1-withclusters.json') # movie features dataset  WITH CLUSTERS!!!

userstats = ratings.groupby('userId', as_index=False).agg({'rating':[np.size,np.mean,np.std,np.min,np.max]}) # statistics on users
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

rr = ratingswmstd['rating']
rmin = ratingswmstd[('rating', 'amin')]
rmax = ratingswmstd[('rating', 'amax')]
ratingswmstd['minmaxrating'] = (rr-rmin)/(rmax-rmin)


user1 = ratingswmstd[ratingswmstd['userId']==1]
user2 = ratingswmstd[ratingswmstd['userId']==2]
user3 = ratingswmstd[ratingswmstd['userId']==3]


movieratings = pd.merge(ratingswmstd, movies, on ='movieId', how='inner')

moviegroups = movieratings.groupby('title')
moviestats = moviegroups.agg({'rating':[np.size,np.mean,np.std]}) # statistics on movies
moviestats[moviestats['rating']['size'] >= 50].sort_values([('rating', 'size')], ascending=False) 





#get movie distribution by users:
userdistr = movieratings.groupby('userId').agg({'rating':[np.size]}).sort_values([('rating', 'size')], ascending=False) 
userdistr.plot.hist(bins=100) # freq of ratings per users
movieratings.rating.plot.hist()
movieratings.title.value_counts().sort_values(ascending = True)[:25]



imdbratings = pd.merge(movieratings, links, on='movieId', how='inner')#.drop(['timestamp',('rating', 'size'), ('rating', 'mean'),  ('rating', 'std'),'title'],1)
imdbratings['binrating'] = np.where(imdbratings['normrating']>=0, 1, -1) # binarized ratings


selectedratings = imdbratings[imdbratings['binrating']==1]
userids = list(set(selectedratings.userId))
userXratings = []
for uid in userids:
    # get all ratings for that userid and put them into a list
    moviesforuid = selectedratings[selectedratings['userId']==uid]
    uxr = {'userid' : uid,
           'countmovies' : moviesforuid.userId.count(),
           'moviesratings' : moviesforuid[['imdbId','title','rating','normrating','timestamp']].to_dict(orient = 'records'),
           'imdbids': list(moviesforuid.imdbId)}
    userXratings.append(uxr)
    
    
    
    
#countmovies = [u['countmovies'] for u  in userXratings] # get the number of movies rated by each user,
# it is used to divide the tfidf values and get the local frequency of the features, so that 
# power users do not dominate, since there are users with 5 movies and some with more than 1000.

# get ratings for each user: still in try phase
#normratingarray = np.array([u['normrating'] for u in userXratings])


def importMoviesfromJSON(JSONfile):
    data = [] # list of movies from json
    with open(JSONfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data

#userXfeatures = importMoviesfromJSON('userXfeatures1') # user feature dataset
# make a merge with the dataset features!!!

for i,ur in enumerate(userXratings):
    curru = ur['userid']
    for j,mr in enumerate(ur['moviesratings']):
        for mf in dataset1:            
            if(mr['imdbId'] == mf['imdbid']):
                userXratings[i]['moviesratings'][j]['cluster'] = mf['movcluster']#[m for m in mf['features'] if '(release)_' not in m]
                

userXfeatures = []
for i,uid in enumerate(userXratings):
    for ff in userXratings[i]['moviesratings']: 
        if('cluster' in ff.keys()):
            userXfeatures.append({'userid':uid['userid'], 'cluster': ff['cluster'], 'imdbid': ff['imdbId'],'rating':ff['rating']})


otherfeatures = userXratings.copy()
userXfeatures2 = []
for i,uid in enumerate(userXratings):
    features = [ff['cluster'] for ff in userXratings[i]['moviesratings'] if('cluster' in ff.keys())]
    uxf = {'userid' : str(uid['userid']),
           'cluster' : features}
    userXfeatures2.append(uxf)


# make a dataframe with all features and userids
userXfeaturesdf2 = pd.DataFrame({'userid' : [u['userid'] for u in userXfeatures2],
                                'imdbids' : [u['imdbids'] for u in userXratings],
                                'clusters' : [u['cluster'] for u in userXfeatures2]})



# make a dataframe with all features and userids
userXfeaturesdf = pd.DataFrame(userXfeatures)
#from the movie clusters and the userXratings ratings, make a new dataset of users
#based on cluster frequency
theusers = userXfeaturesdf.groupby(['cluster','userid'], as_index=False).agg({'rating':[np.size,np.sum,np.mean]})
arr = theusers.pivot(index = 'userid',columns = 'cluster',values = ('rating','size')).fillna(0)
R = arr.as_matrix() # just cluster frequency for each user
R = R/R.sum(axis=1)[:,None] # normalize by sum of all movie ratings per user to get relative freq

#dist_normR = pdist(R,'cosine')

tfidfr = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
Rtfidf = tfidfr.fit_transform(R)
XR = scipy.sparse.csc_matrix(UClust).toarray()
USr = svduc.fit_transform(XR) # scikit returns the U*S matrix



from sklearn.feature_extraction.text import CountVectorizer
countclusterusers = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
UClust = countclusterusers.fit_transform(userXfeaturesdf2.clusters)
import scipy
Xuc = scipy.sparse.csc_matrix(UClust)

from sklearn.decomposition import TruncatedSVD # scikit implementation
# with scikit learn implementation
svduc = TruncatedSVD(n_components=20, n_iter=10, random_state=0)
USuc = svduc.fit_transform(Xuc.toarray()) # scikit returns the U*S matrix
cumulative = np.cumsum(svduc.explained_variance_ratio_)
plt.plot(cumulative, c='blue')
plt.show()

#from bhtsne import tsne
#tsne_Um = tsne(USr)
#plt.scatter(tsne_Um[:, 0], tsne_Um[:, 1], alpha=0.4)



from scipy.spatial.distance import pdist,cosine,squareform
# COSINE METRIC
dist_outusers = pdist(Xuc.toarray(),'cosine')
hclust(dist_outusers,'complete')



    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:24:56 2017

METADATA analysis

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
feature = 'countries'
nodict = {}
#nodict['imdbId'] = [m['imdbid'] for m in metadata if m[feature] == "N/A"]
# use the one below if metadata is already vectorized, else use the above
nodict['imdbId'] = [m['imdbid'] for m in metadata if "N/A" in m[feature]]
nodf = pd.DataFrame.from_dict(nodict)
moviewo = pd.merge(movietitles, nodf, on='imdbId', how='inner')
# inspect the movies without feature
moviewo.head()








def findmovie(imdbid):
    return [m for m in metadata if m['imdbid']==imdbid]

def findmoviefeature(imdbid,feature):
    return [m[feature] for m in metadata if m['imdbid']==imdbid]
nodict['imdbId'] = [m['imdbid'] for m in metadata if "N/A" in m['languages']]
for ii in nodict['imdbId']:
    print(ii)
    print(findmoviefeature(ii,'countries'))







#============================2-TRANSFORM RELEASE DATE======================================
#============================+-ADD MOVIELENS RELEASE DATE WHERE IS N/A=====================
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

# save to JSON! this is the real one!
with open('METADATAVECTORIZED.json', 'w') as fout:
    json.dump(metadata, fout)


#4 metadata inspection
#============================4.1-CLEAN UP TO 100 KEYWORDS-=====================================


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

# save to separate json
with open('METADATAVECTKEYWORD100.json', 'w') as fout:
    json.dump(metadataklim, fout)

#============================4.2-REPLACE LANGUAGE WITH COUNTRY-=====================================

#1 get the country list of movies without language

#2 find the most common language for movies produced in the country of production
#2.1 get frequency of language for each country of production
def createFrquencyTable(word_list):
    #word count
    word_count = {}
    for words in word_list:
        # if word is a list
        for word in words:
            #index is the word
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return word_count

#2.2 get the languages
def findlanguagefor(country):
    import operator
    wordlist = []
    for m in metadataklim:
        # print the first language for a movie with country = country
        if('N/A' not in m['languages'] and country in m['countries']):
            wordlist.append(m['languages'])
    countrylangfreq = createFrquencyTable(wordlist)
    commonlang = list(sorted(countrylangfreq.items(), key=operator.itemgetter(1), reverse=True)[0])[0]
    return commonlang


#replace the N/A of language with the most common language for that country
for i,m in enumerate(metadataklim):
    if('N/A' in m['languages'] and 'N/A' not in m['countries']):
        print(m['imdbid'])
        print(m['languages'])
        print(m['countries'])
        print(findlanguagefor(m['countries'][0]))
        metadataklim[i]['languages'] = findlanguagefor(m['countries'][0])


#============================0-ANALYZE FREQUENCIES-=====================================
def analyzefeature(feature):
    import operator
    elems = []
    for m in metadataklim:
        elems.append(m[feature])
    elemssfreq = list(sorted(createFrquencyTable(elems).items(), key=operator.itemgetter(1), reverse=True))    
    return elemssfreq

# analyze writer:
analyzefeature('writer')[0:20]  

# analyze director:
analyzefeature('director')[0:20] 

# analyze releases:
analyzefeature('release')[0:20] 

# analyze company:
analyzefeature('company')[0:20] 
  
# analyze cast
analyzefeature('cast')[0:20] 

# analyze original music 
analyzefeature('original music')[0:20] 

# analyze keywords
analyzefeature('keywords')[0:20]

# analyze languages
analyzefeature('languages')[0:20]

# analyze countries
analyzefeature('countries')[0:20]

# analyze release
analyzefeature('release')[0:20]

# analyze genre
analyzefeature('genres')[0:20]


#============================5.1-CLEAN movies in original METADATA-=====================================
# REMOVE MOVIES WITH N/A FEATURES (not all):
    # wo countries
    # wo keywords
    # wo director
    # wo release
# keep the movies with N/A:
    # cast
    # company
    # original music
    # writer
metadatacleaned = []
for i,m in enumerate(metadataklim):
    if('N/A' not in m['countries'] and 'N/A' not in m['keywords'] and 
       'N/A' not in m['director'] and 'N/A' not in m['release']):
        metadatacleaned.append(m)

with open('METADATAPRIMARYMISSING.json', 'w') as fout:
    json.dump(metadatacleaned, fout)


#============================5.2-MAKE SEPARATE METADATA WITHOUT the following FEATURES:-=====================================
# remove features with high percentage of N/A:
    # original music 734
    # writer 202
# then proceed as above
metadatawoogw = []
for i,m in enumerate(metadataklim):
    temp = m.copy()
    temp.pop('original music',None)
    temp.pop('writer',None)
    if('N/A' not in temp['countries'] and 'N/A' not in temp['keywords'] and 
       'N/A' not in temp['director'] and 'N/A' not in temp['release']):
        metadatawoogw.append(temp)

with open('METADATANOMUSICANDWRITER.json', 'w') as fout:
    json.dump(metadatawoogw, fout)



    
    
    
#============================5.3-MAKE SEPARATE METADATA WITHOUT N/A-=====================================
# REMOVE MOVIES WITH N/A FEATURES (ALL):
    # wo cast,company,countries,director, writer
    # wo keywords,languages,release, original music
metadatanomissing = []
for i,m in enumerate(metadataklim):
    if(['N/A'] not in m.values()):
        metadatanomissing.append(m)
    
with open('METADATANOMISSING.json', 'w') as fout:
    json.dump(metadatanomissing, fout)










#============================6-EXTRACT THE LIST OF FEATURES FROM EACH DATASET AND MAKE================================== 
#============================ SURE THAT EACH WORD IS LINKED TO THE TYPE OF METADATA=====================================
#============================--THIS IS THE LAST STEP TO GET THE DATASETS--=====================================
def importMoviesfromJSON(JSONfile):
    data = [] # list of movies from json
    with open(JSONfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data

metadatacleaned = importMoviesfromJSON('METADATAPRIMARYMISSING.json')
metadatawoogw = importMoviesfromJSON('METADATANOMUSICANDWRITER.json')
metadatanomissing = importMoviesfromJSON('METADATANOMISSING.json')

f1 = getfeaturedataset(metadatacleaned[0:10])
[f['imdbid'] for f in f1[0:10]]


def getlistoffeatures(d): 
    # d is the dictionary from the metadata files of a single movie
    # need to link the feature to its type (director, company, cast, etc.)
    listoffeatures = []
    for k,v in d.items():
        if(type(v)!=int):
            for item in v:
                listoffeatures.append('('+k+')'+'_'+'('+item+')')
    return listoffeatures # returns the imdbid and the list of features


def getfeaturedataset(data): 
    # for all movies in the dataset data get the features and the corresp imdbid
    finaldata = []
    for m in data:
        curr = {'imdbid' :   m['imdbid'],
                'features' : getlistoffeatures(m)}
        finaldata.append(curr)
    return finaldata



# GET THE FEATURE DATASETS FROM THE 3 METADATA FILES and save to JSON
def getandsavedataset(metadatafile,jsonname):
    finaldata = getfeaturedataset(metadatafile)
    with open(jsonname, 'w') as fout:
        json.dump(finaldata, fout)


getandsavedataset(metadatacleaned,'finaldata1') # dataset 1 movies removed for errors in imdb features but N/A left for other features
getandsavedataset(metadatawoogw,'finaldata2') # dataset 3 as above but without OST and writer (the features with most missing vals in movies)
getandsavedataset(metadatanomissing,'finaldata3') # dataset 3 no missing features






def getdataset(jsonfile):
    data = [] # list of movies from json
    with open(jsonfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data

















        
DE = [m['imdbid'] for m in metadata if 'Hayao Miyazaki' in m['director']]    
len(DE)    

# EXPLORATORY ANALYSIS:
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
moviestats[moviestats['rating']['size'] >= 50].sort_values([('rating', 'size')], ascending=False) 

    
    
    
    
#get movie distribution by users:
userdistr = movieratings.groupby('userId').agg({'rating':[np.size]}).sort_values([('rating', 'size')], ascending=False) 
userdistr.plot.hist(bins=100) # freq of ratings per users
movieratings.rating.plot.hist()
movieratings.title.value_counts().sort_values(ascending = True)[:25]


imdbratings = pd.merge(movieratings, links, on='movieId', how='inner')
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






# get the dataset and begin to make the ITEMxFEATURE dataset
dataset1 = getdataset('finaldata1')





# make the userXitemfeature dataset:
imdbratings = pd.merge(movieratings, links, on='movieId', how='inner').drop(['timestamp',('rating', 'size'), ('rating', 'mean'),  ('rating', 'std'),'title'],1)
selectedratings = imdbratings[imdbratings['normrating']>=0]
userids = list(set(selectedratings.userId))
userXratings = []
for uid in userids:
    # get all ratings for that userid and put them into a list
    moviesforuid = selectedratings[selectedratings['userId']==uid]
    uxr = {'userid' : uid,
           'moviesratings' : moviesforuid[['imdbId','rating']].to_dict(orient = 'records')}
    userXratings.append(uxr)
    

# make a merge with the dataset features!!!
for i,ur in enumerate(userXratings):
    curru = ur['userid']
    for j,mr in enumerate(ur['moviesratings']):
        for mf in dataset1:            
            if(mr['imdbId'] == mf['imdbid']):
                userXratings[i]['moviesratings'][j]['features'] = mf['features']


# the dataset of user features!!!!
userXfeatures = []
for i,uid in enumerate(userXratings):
    features = [ff['features'] for ff in userXratings[i]['moviesratings'] if('features' in ff.keys())]
    uxf = {'userid' : uid['userid'],
           'features' : [item for sublist in features for item in sublist]}
    userXfeatures.append(uxf)
    

# save the user features dataset to JSON!!



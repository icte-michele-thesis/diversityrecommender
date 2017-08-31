#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:39:16 2017

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


dataset1 = getdataset('finaldata1') # movie features dataset

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
    
countmovies = [u['countmovies'] for u  in userXratings] # get the number of movies rated by each user,
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
dataset1 = getdataset('finaldata1')
for i,ur in enumerate(userXratings):
    curru = ur['userid']
    for j,mr in enumerate(ur['moviesratings']):
        for mf in dataset1:            
            if(mr['imdbId'] == mf['imdbid']):
                userXratings[i]['moviesratings'][j]['features'] = mf['features']#[m for m in mf['features'] if '(release)_' not in m]
             #       

# the dataset of user features!!!!
userXfeatures = []
for i,uid in enumerate(userXratings):
    features = [ff['features'] for ff in userXratings[i]['moviesratings'] if('features' in ff.keys())]
    uxf = {'userid' : str(uid['userid']),
           'features' : [item for sublist in features for item in sublist]}
    userXfeatures.append(uxf)



#for i,uf in enumerate(userXfeatures):
#    stringfeature = uf['features']
#    strf = ''
#    for s in stringfeature:
#        strf += ' '+s
#    userXfeatures[i]['string_features'] = strf

featuresnorelease = []
for uf in [u['features'] for u in userXfeatures]:
    featuresnorelease.append([f for f in uf if '(release)_' not in f])# and '(genres)' in f])

# make a dataframe with all features and userids
userXfeaturesdf = pd.DataFrame({'userid' : [u['userid'] for u in userXfeatures],
                                'imdbids' : [u['imdbids'] for u in userXratings],
                                'features' : [u['features'] for u in userXfeatures]})


def wordclouduser(u):
    from wordcloud import WordCloud
    user0 = ''
    for f in userXfeatures[u]['features']:
        user0 += f+' '
    wc = WordCloud().generate(user0)
    
    import matplotlib.pyplot as plt
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


## get all movies containing that feature for a given userid and weight that feature
#
#
#
#def isfeatureinmovie(feature,imdbid):
#    # requires the finaldata1 dataset in dict 
#    return None
#

# ====== CUSTOM TFIDF
import math

#def term_frequency(term, tokenized_document):
#    # normal tf
#    return tokenized_document.count(term)

  
#def adjusted_term_frequency(term, tokenized_document):
#    # tf normalization to remove long-document bias
#    return tokenized_document.count(term)/len(tokenized_document) # the percentage of that term in the document
#
#def augmented_term_frequency(term, tokenized_document):
#    # other bias removal tf normalization
#    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
#    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))
#
#def tf_rating_weighted(term, tokenized_document):
#    tf = adjusted_term_frequency(term, tokenized_document)
#    rw = term_rating_weight(term, tokenized_document, curru_movies,currfeatures)
#    return tf*rw


#def inverse_document_frequencies(tokenized_documents):
#    idf_values = {}
#    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
#    for tkn in all_tokens_set:
#        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
#        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
#    return idf_values

#def n_containing(term, tokenized_documents):
#    return sum(1 for blob in tokenized_documents if term in blob)
#
#def idf(term, tokenized_documents):
#    return math.log(len(tokenized_documents) / (1 + n_containing(term, tokenized_documents)))
#
#def idfs(tokenized_documents):
#    idf_values = {}
#    for i,doc in enumerate(tokenized_documents):
#        idf_values = {term: idf(term, tokenized_documents) for term in doc}
#    return idf_values
#
#
#def tfidf(tokenized_documents): # all lists of features for all users
#    idf = inverse_document_frequencies(tokenized_documents)
#    tfidf_documents = []
#    for document in tokenized_documents:
#        doc_tfidf = []
#        for term in idf.keys():
#            tf = sublinear_term_frequency(term, document)
#            doc_tfidf.append(tf * idf[term])
#        tfidf_documents.append(doc_tfidf)
#    return tfidf_documents





## try match user imdbids and dataset features
#curru_movies = userXratings[0]['moviesratings']
#curruimdbids = [int(c['imdbId']) for c in curru_movies]
## match with dataset1
#currumovies = [i for i, j in zip(curruimdbids, dataset1) if i == j['imdbid']]
#currfeatures = []
#for i in curruimdbids:
#    currf = [f['features'] for f in dataset1 if f['imdbid'] == i][0]
#    currfeatures.append(currf)
#    
#    
#fexample = '(company)_(Walt Disney Productions)'
#featmatch = [i for i,m in enumerate(currfeatures) if fexample in m]
#
#featratings = [curru_movies[i]['rating'] for i in featmatch]
#avgratings = sum(featratings)/float(len(featratings))






def term_rating_weight(term, tokenized_document, curru_movies,currfeatures):
    matchterm = [i for i,m in enumerate(currfeatures) if term in m]
    featratings = [curru_movies[i]['rating'] for i in matchterm]
    return sum(featratings)/float(len(featratings)),sum(featratings),1 + math.log(sum(featratings))

def matchusermoviefeatures(curruser): # the index of user
    # match the imdbids and the movie features dataset
    # to be called once per user
    curru_movies = userXratings[curruser]['moviesratings']
    curruimdbids = [int(c['imdbId']) for c in curru_movies]
    # intersect imdbids with the effectively present in dataset1, since some movies were deleted from here
    alldataset1imdbids = set([f['imdbid'] for f in dataset1]).intersection(curruimdbids)
    currfeatures = []
    for i in alldataset1imdbids:
        release_ = set([])
        currf = set([f['features'] for f in dataset1 if f['imdbid'] == i][0])
        currf = [cf for cf in list(currf) if '(release)' not in cf]
        currfeatures.append(currf)
    return curru_movies, currfeatures #return movies rated and features for each movie
  


def term_rating_weight(term, curru_movies, currfeatures):
    matchterm = [i for i,m in enumerate(currfeatures) if term in m]
    featratings = [curru_movies[i]['rating'] for i in matchterm]
    return sum(featratings)/float(len(featratings)),sum(featratings),1 + math.log(sum(featratings)),len(featratings)/float(len(curru_movies))


#newrfreqs = []
#for f in list(usernnz[1]):
#    currf = feature_array[f]
#    newrfreqs.append(term_rating_weight(currf,umfeatures[0],umfeatures[1]))

feature_array = np.array(v.get_feature_names())

userfreqs = []
for u in range(len(userids)):
    usernnz = np.nonzero(UF[u,:]!=0) # the indexes of each feature present in user u
    umfeatures = matchusermoviefeatures(u)
    newrfreqs = []
    for f in list(usernnz[1]): # for each feature (index)
        currfeature = feature_array[f] # we get the feature name
    # then extract the ratings for that user,feature pair
        currfeatindex = {'rating_frequencies':term_rating_weight(currfeature,umfeatures[0],umfeatures[1]),
                         'index':f}
        newrfreqs.append(currfeatindex)
    userfreqs.append(newrfreqs)


UFrw = UF.copy() # uses the average rating to weight terms
UFrw2 = UFrw.copy() # uses the sum of all ratings
UFrw3 = UFrw.copy() # uses a sublinear weight 
UFrw4 = UFrw.copy() # uses the percentage of movies containing a that movie feature and multiplies it with the average rating of those movies
UFrw5 = UFrw.copy() # uses the percentage of movies containing a that movie feature
for u,user in enumerate(userfreqs):
    for feature in user:
        f = feature['index']
        UFrw[u,f] *= feature['rating_frequencies'] [0]
        UFrw2[u,f] *= feature['rating_frequencies'][1]
        UFrw3[u,f] *= feature['rating_frequencies'][2] 
        UFrw4[u,f] = feature['rating_frequencies'][3]*feature['rating_frequencies'] [0]
        UFrw5[u,f] *= feature['rating_frequencies'][3] 




u = 0
usernnz = np.nonzero(UF[u,:]!=0) # the indexes of each feature present in user u
umfeatures = matchusermoviefeatures(u)
for f in usernnz[1][5915]:
    print(UF[u,f]!=0)
usernnzsort = np.argsort(usernnz[0])[::-1]

[x for x in feature_array if 'release' in x][:10]
l = UF[0,:].toarray()
l = (l*-1).argsort()
companies = [feature_array[x] for x in l if 'release)_' in feature_array[x]][:10]

topfeatures = []
for l in UF.toarray():
    topfeatures.append(gettopfeaturesforuser(l))

def gettopfeaturesforuser(l):
    co = [feature_array[x] for x in (l*-1).argsort() if 'company)_' in feature_array[x]][:10]
    d = [feature_array[x] for x in (l*-1).argsort() if 'director)_' in feature_array[x]][:10]
    k = [feature_array[x] for x in (l*-1).argsort() if 'keywords)_' in feature_array[x]][:10]
    g = [feature_array[x] for x in (l*-1).argsort() if 'genres)_' in feature_array[x]][:10]
    cc = [feature_array[x] for x in (l*-1).argsort() if 'countries)_' in feature_array[x]][:10]
    w = [feature_array[x] for x in (l*-1).argsort() if 'writer)_' in feature_array[x]][:10]
    ll = [feature_array[x] for x in (l*-1).argsort() if 'languages)_' in feature_array[x]][:10]
    ca = [feature_array[x] for x in (l*-1).argsort() if 'cast)_' in feature_array[x]][:10]
    r = [feature_array[x] for x in (l*-1).argsort() if 'release)_(' in feature_array[x]][:10]    
    o = [feature_array[x] for x in (l*-1).argsort() if 'original music)_' in feature_array[x]][:10]    
    
    topf = {'company':co,
            'director':d,
            'keywords':k,
            'genres':g,
            'countries':cc,
            'writer':w,
            'languages':ll,
            'cast':ca,
            'release':r,
            'original music':o
            }
    return topf
    

#
# perform only countvectorization, instead of tf-idf
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=1.0, min_df=5, tokenizer=lambda i:i, lowercase=False)
UFcv = cv.fit_transform(userXfeaturesdf['imdbids'])

Xcv = scipy.sparse.csc_matrix(UFcv).asfptype()
Ucv, Scv, Vtcv = svds(Xcv, 300)
US_ordcv = Ucv[:,::-1]*(Scv[::-1])#**2)
    
tsne_U = tsne(US_ordcv)
# t-sne plot
plt.scatter(US_ordcv[:, 0], US_ordcv[:, 1], alpha=0.2)

# svd 670 plot
plt.figure()
plt.scatter(US_ordcv[:, 0], US_ordcv[:, 1], alpha=0.1)
plt.show()

ax = Axes3D(plt.figure())
ax.scatter(US_ordcv[:, 0], US_ordcv[:, 1], US_ordcv[:, 2], alpha=0.6)
plt.show()
jaccardUFcv = pdist(Xcv.toarray(),'jaccard')
hclust(jaccardUFcv,'ward')





from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_df=1.0, min_df=5, tokenizer=lambda i:i, 
                    sublinear_tf=True, lowercase=False,use_idf=True)
UF = v.fit_transform(userXfeaturesdf['features'])
# divide the UF rows by the number of movies rated:
#UFnorm = UF / np.array(countmovies)[: , None] # this improves the svd screeplot but the sv are very small


from scipy.sparse.linalg import svds
import scipy

X = scipy.sparse.csc_matrix(UF)


# non negative matrix factorization
from sklearn.decomposition import NMF
nmfmodel = NMF(n_components=10, init='nndsvd', random_state=1,alpha=.1, l1_ratio=.1)
W = nmfmodel.fit_transform(X)
H = nmfmodel.components_
nmfmodel.reconstruction_err_


X1 = scipy.sparse.csc_matrix(UFrw)
#X2 = scipy.sparse.csc_matrix(UFrw2)
X3 = scipy.sparse.csc_matrix(UFrw3)
X4 = scipy.sparse.csc_matrix(UFrw4)
X5 = scipy.sparse.csc_matrix(UFrw5)
U, S, Vt = svds(X, 50)
U1, S1, Vt1 = svds(X1, 50)
U2, S2, Vt2 = svds(X2, 50)
U3, S3, V3 = svds(X3, 50)
U4, S4, V4 = svds(X4, 100)
U5, S5, V5 = svds(X5, 100)
#cumulative = np.cumsum(np.power(sorted(S, reverse=True),2)/sum(np.power(sorted(S, reverse=True),2)))
cumulative = np.cumsum(S[::-1]**2/sum(S[::-1]**2))
plt.plot(cumulative, c='blue')
plt.show() # at around 200 singular values the shit is high
plt.plot(S[::-1],c='blue')


last = S[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last)
acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration)
plt.show()
k = acceleration.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

Vtord = Vt[::-1,:]
plt.scatter(Vtord[2, :], Vtord[1, :], alpha=0.1)
plt.show()

# scree plot
def screeplot(S):
    """ works for the full Sigma """
    #eigvals = np.power(sorted(S, reverse=True),2) / np.cumsum(S)[-1]
    eigvals = S[::-1]**2/sum(S)
    numvars = min(X.shape)-1
    fig = plt.figure(figsize=(8,5))
    sing_vals = np.arange(numvars) + 1
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

# we want to use the U*S matrix instead of the single U, because we need more
# diversification power between users given by the magnitude of the singular values.

US_ord = U[:,::-1]*(S[::-1]**2) # the U*S flipped by descending sing val order
#Uord = U[:,::-1] # the plot is the same but on a very small scale!!!


from bhtsne import tsne
tsne_U = tsne(US_ord)
# t-sne plot
plt.scatter(tsne_U[:, 0], tsne_U[:, 1], alpha=0.2)

# svd 670 plot
plt.figure()
plt.scatter(US_ord[:, 0], US_ord[:, 1], alpha=0.1)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure())
ax.scatter(US_ord[:, 0], US_ord[:, 1], US_ord[:, 2], alpha=0.6)
plt.show()




def scaleusord(US_ord):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(US_ord)

usordmmx = scaleusord(US_ord)


US_ord1 = U1[:,::-1]*S1[::-1] # the U*S flipped by descending sing val order
US_ord2 = U2[:,::-1]*S2[::-1] # the U*S flipped by descending sing val order
US_ord3 = U3[:,::-1]*S3[::-1] # the U*S flipped by descending sing val order










# get the similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,squareform
# metrics  [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’]
dist_out = pdist(US_ord,'cosine')   #pairwise_distances(US_ord, metric='manhattan')
dist_out = 1-pairwise_distances(US_ord, metric='cosine')
dist_out_nosvd = 1-pairwise_distances(UF, metric='cosine')
#plt.pcolor(dist_out,cmap=plt.cm.Reds)
#plt.colorbar()

n = 3
u0 = userXratings[n]['moviesratings']
utry1 = list(dist_out[n,:])
utry1no = list(dist_out_nosvd[n,:])
# the most similar user to 0:
top10 = sorted(range(len(utry1)), key=lambda i: utry1[i])[-10:]
top10no = sorted(range(len(utry1no)), key=lambda i: utry1no[i])[-10:]
#mostsimusers = utry1.index(top10) # 232 or 126
plt.scatter(US_ord[:, 0], US_ord[:, 1], alpha=0.1)
plt.scatter(US_ord[n, 0], US_ord[n, 1], alpha=0.8)
for x in top10[:-1]:
    plt.scatter(US_ord[top10[:-1], 0], US_ord[top10[:-1], 1], alpha=0.8)
plt.show()
plt.figure()
plt.scatter(tsne_U[:, 0], tsne_U[:, 1], alpha=0.2)
plt.scatter(tsne_U[n, 0], tsne_U[n, 1], alpha=0.8)
for x in top10[:-1]:
    plt.scatter(tsne_U[top10[:-1], 0], tsne_U[top10[:-1], 1], alpha=0.8)
plt.show()




feature_array = np.array(v.get_feature_names())
tfidf_sorting = np.argsort(UF[n,:].toarray())[::-1]
toptermsforuser = feature_array[tfidf_sorting][:10]


# try the intersection cosine and jaccard with the tfidf matrix
a1 = np.array(UF[u1,:])
a2 = np.array(UF[u2,:])
out = intersectfeatures(0,3)

#dout = 1-pairwise_distances(a1,a2, metric='manhattan')

u1 = 0
u2 = 1
def cosinesimbetweenintersect(u1,u2):
    from scipy.spatial.distance import cosine
    out = intersectfeatures(u1,u2)
    a1 = out[0]
    a2 = out[1]
    return 1-cosine(a1[out[2]],a2[out[2]])


def intersectfeatures(u1,u2):
    a1 = np.array(X[u1,:].toarray())
    a2 = np.array(X[u2,:].toarray())
    return a1,a2,np.nonzero((a1!=0) & (a2!=0))


listcossim0 = []
for i in range(0,len(userids)):
    listcossim0.append(cosinesimbetweenintersect(0,i))

# the most similar user to 0:
top10 = sorted(range(len(listcossim0)), key=lambda i: listcossim0[i])[-10:]
mostsimuser = listcossim0.index(max(listcossim0)) # 232 or 126

# find the movies rated by both
u0 = userXratings[0]['moviesratings']
u126 = userXratings[126]['moviesratings']

#441 and 267 
def moviesincommon(u1,u2):
    u1m = [m['imdbId'] for m in userXratings[u1]['moviesratings']]
    u2m = [m['imdbId'] for m in userXratings[u2]['moviesratings']]
    return set(u1m).intersection(u2m),len(set(u1m).intersection(u2m))/len(set(u1m).union(u2m)),len(u1m)/len(u2m)



def jaccarduserfeatures(u1,u2):
    u1m = userXfeatures[u1]['features']
    u2m = userXfeatures[u2]['features']
    return set(u1m).intersection(u2m),len(set(u1m).intersection(u2m))/len(set(u1m).union(u2m))

jsu0 = []
n = 4
for u in range(0,len(userids)):
    if(u!=4):
        jsu0.append(jaccarduserfeatures(n,u)[1])
k = np.argmax(jsu0)
jsu = jaccarduserfeatures(n,k)

jaccard_out = np.zeros((len(userids),len(userids)))
for ui in range(0,len(userids)):
    for uj in range(0,len(userids)):
        if ui!=uj:
            jaccard_out[ui][uj] = jaccarduserfeatures(ui,uj)[1]

# jaccard from dataframe userXfeaturesdf
jaccard_out = pdist(userXfeaturesdf['features'], 'jaccard')
jaccard_out1 = list(jaccard_out[3,:])
jo = np.argmax(jaccard_out1)



# CLUSTERING!!!!!!
# import
from scipy.cluster.hierarchy import dendrogram, linkage
dist_out = pdist(US_ord, 'cityblock')#,'cityblock')
dist_outw = pdist(W,'cosine')#,'cityblock')
Z = linkage(dist_outw,'average')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, dist_outw)
c
plotdendrogram(Z)
# plot the dendogram
def plotdendrogram(Z):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram: average-cityblock')
    plt.xlabel('users')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=7.,  # font size for the x axis labels
        
    )
    plt.show()

dist_outw = pdist(W)
hclust(dist_outw,'average')






# Compute and plot first dendrogram.
def hclust(dist_out,method):
    fig = plt.figure(figsize=(8,8))
    # x ywidth height
    ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
    Y = linkage(dist_out, method=method)
    Z1 = dendrogram(Y, orientation='left',leaf_font_size=3.) # adding/removing the axes
    ax1.set_xticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = squareform(dist_out)
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)










# select the most suitable number of clusters
last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print( "clusters:"+ k)







from scipy.cluster.vq import kmeans,vq

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(US_ord,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()











"""
how to compute user similarity:
    get the matrix USigma from SVD and create a dissimilarity matrix out of it.
    cluster the users based on hierarchical clustering or kmeans
how to find similar users:
    for a given user u in cluster c, (in order to pick relevant users)
    find find the least N similar users to u and extract their movies, if not rated by u

how to get the movie list:
    for each dissimilar user to u, pick M movies not rated by u
    then compare the movies to those rated by u using the similarity index from CBF (LSA)
how to get a diverse movie list for u:
    inspect movies rated well by u, then
    for each user picked v!=u, get the union of all movies so that:
        U(v,i)\I(u): the union of all movies of all picked users v,
                     without the movies rated by u.
    order the movie set by the avg distance to the history of u and pick the topN.




given the user u profile heterogeneity, 
calculated as the avg similarity btw movies rated well,
find the topN elements that are diverse and relevant to u



"""
















#                                                    # gensim LSA of user profiles
## GENSIM
#import gensim
#from gensim import corpora, similarities, models
#from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
#np.random.seed(42)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
#
#userfeaturetexts = [f['features'] for f in userXfeatures]
#dictionary = corpora.Dictionary(userfeaturetexts) #117897 unique keywords
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in userfeaturetexts]
#
#corpora.MmCorpus.serialize('user_features_1_corpus',doc_term_matrix)
##LSA_IMDB_corpus = corpora.MmCorpus('LSA_IMDB_CORPUS')
#
## building the TFIDF matrix from the docterm matrix
#tfidf = gensim.models.TfidfModel(doc_term_matrix)
#tfidf_matrix = tfidf[doc_term_matrix]
#
## LDA training
#lsi = gensim.models.lsimodel.LsiModel(corpus=tfidf_matrix, id2word=dictionary, num_topics=400)
#corpus_lsi = lsi[corpus_tfidf]
#
## print the topics
#lsi.print_topics(50)
#































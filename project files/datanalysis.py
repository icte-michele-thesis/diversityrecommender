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











"""
    #==============================     BEGINNING OF
    #==============================    LSA WITH GENSIM, THE CONTENT BASED FILTERING METHOD THAT WILL BE USED
    #==============================    IN THE EXPERIMENT
    #==============================    
"""

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


def gettitles():
    all_links = pd.read_csv("ml-latest-small/links.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    movietitles = pd.merge(all_links, movies, on ='movieId', how='inner').drop(['movieId','tmdbId'],1)
    return movietitles.to_dict(orient = 'records')



def importMoviesfromJSON(JSONfile):
    data = [] # list of movies from json
    with open(JSONfile, encoding='utf-8') as datafile:
        data = json.load(datafile)
    return data

finaldata1 = importMoviesfromJSON('finaldata1')
finaldata2 = importMoviesfromJSON('finaldata2')
finaldata3 = importMoviesfromJSON('finaldata3')




    #==============================    1) PREPARE DICTIONARIES AND TFIDF MATRICES for finaldata1

    
features1 = finaldata1copydf.features_tok

ssss = [i for i in features1 if '(languages)_(l)' in i]

# building the dictionary and the docterm matrix

f1_dictionary = corpora.Dictionary( features1)
f1_doc_term_matrix = [f1_dictionary.doc2bow(doc) for doc in features1]
                            
corpora.MmCorpus.serialize('LSA_finaldata1_CORPUS_full',f1_doc_term_matrix)
#LSA_IMDB_corpus = corpora.MmCorpus('LSA_IMDB_CORPUS')

# building the TFIDF matrix out of the docterm matrix
f1_tfidf = gensim.models.TfidfModel(f1_doc_term_matrix)
f1_tfidf_matrix = f1_tfidf[f1_doc_term_matrix]


    #==============================     2) APPLY LSA ON finaldata1
# prepare the model
f1_lsi = gensim.models.LsiModel(f1_tfidf_matrix, id2word=f1_dictionary, num_topics=40)

# train the model on all plots
f1_corpus_lsi = f1_lsi[f1_tfidf_matrix]

#LSA_IMDB_lsi.print_topics(50)

# index the plot similarities
f1_index = similarities.MatrixSimilarity(f1_corpus_lsi)#LSA_IMDB_lsi[LSA_IMDB_corpus_lsi])
f1_index.save('f1_INDEX_full.index')


    #==============================     3) PREPARE SIMILARITY QUERIES FOR GIVEN IMDBIDS


def get_test_movie(imdbid_test,finaldata):
    """ it returns the bag of words corresponging to the plot of a given movie """
    testfeatures = [f for f in finaldata if f['imdbid'] == imdbid_test]
    features = testfeatures[0]['features']
    features_bow = f1_dictionary.doc2bow(features) # dictionary is can be a parameter to this function
    return features_bow # test_IMDB_bow,test_WIKI_bow
    


# test on IMDB LSA
def get_IMDB_LSA_similaritylist(imdbid):
#    start = time.time()
    test_feature_bow = get_test_movie(imdbid,finaldata1)
    tfidf_bow = f1_tfidf[test_feature_bow] #  tfidf can be a parameter
    vec_lsi = f1_lsi[tfidf_bow] #             lsi can be a parameter
    sims_imdb_lsa = f1_index[vec_lsi]
    sims_imdb_lsa = sorted(enumerate(sims_imdb_lsa), key=lambda item: -item[1])
#    end = time.time()
#    print("Time elapsed: "+ str(end - start))
    return sims_imdb_lsa


def getlistofimdbids(imdbid):
    similarto = get_IMDB_LSA_similaritylist(imdbid)
    indexes = [f[0] for f in similarto]
    fin1 = np.array(finaldata1.copy())
    return [f['imdbid'] for f in list(fin1[indexes])]
# test imdbids:
#   245429 spirited away
#   133093 matrix
  
def getsimilartitles(imdbid,topn):    
    imdbidsofsims = getlistofimdbids(imdbid)[:topn]
    similarmovies = []
    movietitles = gettitles()
    for imdbid in imdbidsofsims:
        similarmovies.append([d['title'] for d in movietitles if d['imdbId']==imdbid][0])
    return similarmovies

"""
    #==============================     END OF:
    #==============================    LSA WITH GENSIM, THE CONTENT BASED FILTERING METHOD THAT WILL BE USED
    #==============================    IN THE EXPERIMENT
    #==============================    
"""

# try some for a given movie id
similars = getsimilartitles(2488496,10)
get_IMDB_LSA_similaritylist(2488496)

def getsimilarityfrommatrix(imdbid,topn,distmatrix):
    imdbindex = allimdbids.index(imdbid)
    l1 = list(sorted(distmatrix[imdbindex,:]))
    l2 = list(np.argsort(distmatrix[imdbindex,:]))
    l12 = list(zip(l2,l1))[:topn]
    indexes = [f[0] for f in l12]
    fin1 = np.array(finaldata1.copy())
    imdbidsofsims = [f['imdbid'] for f in list(fin1[indexes])]
    similarmovies = []
    movietitles = gettitles()
    for imdbid in imdbidsofsims:
        similarmovies.append([d['title'] for d in movietitles if d['imdbId']==imdbid][0])
    return similarmovies,np.average(np.array(l1[:topn]))




"""
    #==============================     beginning of
    #==============================    CREATION OF USER MATRIX, FEATURE MATRIX AND USER FEATURE MATRIX
    #==============================    
"""









finaldata1copy = finaldata1.copy()

for i,uf in enumerate(finaldata1copy):
    stringfeature = uf['features']
    strf = ''
    for s in stringfeature:
        strf += ' '+s
    finaldata1copy[i]['string_features'] = strf


#keywordfeatures = []
#kftokenized = []
#for i,uf in enumerate(finaldata1copy):
#    stringfeature = uf['features']
#    keywordfeat = [f for f in stringfeature if 'keyword' in f]
#    kftokenized.append(keywordfeat)
#    strf = ''
#    for s in keywordfeat:
#        strf += ' '+s
#    keywordfeatures.append(strf)



mtitles = gettitles()
titles = []
for m in finaldata1copy:
    titles.append([t['title'] for t in mtitles if t['imdbId']==m['imdbid']][0])
    
featuresnodecade = []
for uf in [u['features'] for u in finaldata1copy]:
    featuresnodecade.append([f for f in uf if '(decade)_' not in f])# and '(keywords)' in f])

finaldata1copydf = pd.DataFrame({'imdbid' : [u['imdbid'] for u in finaldata1copy],
                                 'title' : [t for t in titles],
                                #'features' : [u['string_features'] for u in finaldata1copy],
                                'features_tok' : [u for u in featuresnodecade],
                                #'keywordfeatures' : [k for k in keywordfeatures]
                                })

allimdbids = [f['imdbid'] for f in finaldata1copy]




# perform only countvectorization, instead of tf-idf
from sklearn.feature_extraction.text import CountVectorizer
cv = TfidfVectorizer(max_df=1.0, min_df=1, tokenizer=lambda i:i, lowercase=False)
UFcv = cv.fit_transform(finaldata1copydf.features_tok)

Xcv = scipy.sparse.csc_matrix(UFcv).asfptype()
Ucv, Scv, Vtcv = svds(Xcv, 300)
US_ordcv = Ucv[:,::-1]*(Scv[::-1])
    
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

distanceUFcv = pdist(US_ordcv,'euclidean')

US_ordcv = Ucv[:,::-1]*(Scv[::-1])
distanceUFcv = pairwise_distances(US_ordcv,metric='cosine')
getsimilarityfrommatrix(2488496,20,distanceUFcv)
#hclust(jaccardUFcv,'ward')

Z = linkage(jaccardUFcv,'ward')#**(1/2),'ward')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, jaccardUFcv)#**(1/2))
c
# plot the dendogram
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












# perform TFIDF on the dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
vmf = TfidfVectorizer(max_df=0.8, min_df=20, tokenizer=lambda i:i, lowercase=False)
UFmf = vmf.fit_transform(finaldata1copydf.features_tok)
UFmf.shape
# get the global frequency of terms 
feature_array = np.array(vmf.get_feature_names())
tfidf_sorting = np.argsort(UFmf.toarray()).flatten()[::-1]
list(feature_array[tfidf_sorting][:40])
# analyze frequencies
listoffreqs = sorted(UFmf.sum(axis = 0).tolist()[0], reverse=True)
plt.plot(listoffreqs)


from sklearn.decomposition import TruncatedSVD
svdmf = TruncatedSVD(n_components=200, n_iter=10, random_state=0)
UmSm = svdmf.fit_transform(Xmf) 
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())
cumulative = np.cumsum(svdmf.explained_variance_ratio_)
plt.plot(cumulative, c='blue')
plt.show()


from scipy.sparse.linalg import svds
import scipy

Xmf = scipy.sparse.csc_matrix(UFmf)
Um, Sm, Vtm = svds(Xmf, 100) #svds(Xmf, min(Xmf.shape)-1)
#from scipy import dot, linalg
#USsem = dot(Um,linalg.diagsvd(Sm, 100, 100))

cumulative = np.cumsum(np.power(sorted(Sm, reverse=True),2)/sum(np.power(sorted(Sm, reverse=True),2)))
plt.plot(cumulative, c='blue')
plt.show()
plt.plot(sorted(Sm, reverse=True))

# get the second derivative of the singular values to extract the optimal elbow point:
dx = np.diff(Sm, axis = 0)
dx_first = np.diff(dx, axis = 0)

#Sminv = np.linalg.inv(np.diag(Sm[::-1]))
#
#USigmam = np.matmul(Um[:,::-1],Sminv)#Um[:,::-1]*(Sminv)
Usigma = Um[:,::-1]*(Sm[::-1])
plt.scatter(Usigma[:, 0], Usigma[:, 1], alpha=0.1)


dist_out = pdist(Xmf.toarray(),'cosine')
#hclust(dist_out,'ward')
Z = linkage(dist_out,'average')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, dist_out)
c
# plot the dendogram
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



# t-sne visualization
from bhtsne import tsne
tsne_Um = tsne(UmSmconc)
plt.scatter(tsne_Um[:, 0], tsne_Um[:, 1], alpha=0.1)

# t-sne plot
maxplot = 600
plt.figure()
#plt.scatter(tsne_Um[:, 0], tsne_Um[:, 1], alpha=0.1)
plt.scatter(tsne_Um[-maxplot:, 0], tsne_Um[-maxplot:, 1], alpha=0.1)
for i, txt in enumerate(titles[-maxplot:]):
    #plt.annotate(txt, (tsne_Um[i,0],tsne_Um[i,1]))
    plt.text(tsne_Um[i, 0], tsne_Um[i, 1], s=txt,
             horizontalalignment='center', fontsize=5)
plt.show()


# do a test plot with random movies:
from random import randint
plt.figure()
for i in range(0,100):
    c = randint(7000,len(titles))
    plt.scatter(USigmam[c, 0], USigmam[c, 1], alpha=0.1)
    plt.text(USigmam[c, 0], USigmam[c, 1], s=titles[c],
             horizontalalignment='center', fontsize=6)
plt.show()

# svd 670 plot
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure())
ax.scatter(USigmam[:, 0], USigmam[:, 1], USigmam[:, 2], alpha=0.1)
plt.show()




def knntsneplot(imdbid,topn):
    """ a function that returns the coordinates for tsne and svd of the 
        most topn similar movies to a given one"""
    similarimdbids = getlistofimdbids(imdbid)[:topn] # get the imdbids
    mtitles = getsimilartitles(imdbid,topn)# get the titles to label them
    # get the indexes of elements
    indexessimilar = []
    for si in similarimdbids:
        currind = allimdbids.index(si)
        indexessimilar.append(currind)
    subtsneX = []
    subtsneY = []  
    subsvdX = []
    subsvdY = []
    for idxs in indexessimilar:
        subtsneX.append(tsne_Um[idxs, 0])
        subtsneY.append(tsne_Um[idxs, 1])
        
        subsvdX.append(USigmam[idxs, 0])
        subsvdY.append(USigmam[idxs, 1])
    
    return subtsneX, subtsneY, subsvdX, subsvdY, mtitles

# plot the most similar movies to a given one with tsne coordinates
# imdbid test 245429, 133093, 120737
tsneX, tsneY, svdX, svdY, ttls = knntsneplot(120737,90)
plt.figure()
plt.scatter(tsneX, tsneY, alpha=0.1)
for i, txt in enumerate(ttls):
    #plt.annotate(txt, (tsne_Um[i,0],tsne_Um[i,1]))
    plt.text(tsneX[i], tsneY[i], s=txt,
             horizontalalignment='center', fontsize=6)
plt.show()

# plot the most similar movies with svd coordinates
plt.figure()
plt.scatter(svdX, svdY, alpha=0.1)
for i, txt in enumerate(ttls):
    #plt.annotate(txt, (tsne_Um[i,0],tsne_Um[i,1]))
    plt.text(svdX[i], svdY[i], s=txt,
             horizontalalignment='center', fontsize=6)
plt.show()





from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,squareform

from scipy.cluster.hierarchy import dendrogram, linkage
dist_outm = pdist(USigmam,'cityblock')
Zm = linkage(dist_outm,'average')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

cm, coph_distsm = cophenet(Zm, dist_outm)
cm
# plot the dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram: average-cityblock')
plt.xlabel('users')
plt.ylabel('distance')
dendrogram(
    Zm,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=3.,  # font size for the x axis labels
    
)
plt.show()














from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

from scipy.spatial.distance import cdist, pdist
K = range(1,100)
KM = [KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(USmf) for k in kk]
centroids = [k.cluster_centers_ for k in KM]
D_k = [cdist(USmf, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/USmf.shape[0] for d in dist]
# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(USmf)**2)/USmf.shape[0]
bss = tss-wcss
varExplained = bss/tss*100
kIdx = 10-1
##### plot ###
kIdx = 2
# elbow curve
# Set the size of the plot
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.subplot(1, 2, 2)
plt.plot(K, varExplained, 'b*-')
plt.plot(K[kIdx], varExplained[kIdx], marker='o', markersize=12,
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.tight_layout()





# tfidf + svd + single k-means:

featuresnodecade = []
for uf in [u['features'] for u in finaldata1copy]:
    featuresnodecade.append([f for f in uf if '(decade)_' not in f and '(keywords)' in f])

finaldata1copydf = pd.DataFrame({'imdbid' : [u['imdbid'] for u in finaldata1copy],
                                 'title' : [t for t in titles],
                                #'features' : [u['string_features'] for u in finaldata1copy],
                                'features_tok' : [u for u in featuresnodecade],
                                #'keywordfeatures' : [k for k in keywordfeatures]
                                })


cvmf = CountVectorizer(max_df=0.8, min_df=30, tokenizer=lambda i:i, lowercase=False)
UFcmf = cvmf.fit_transform(finaldata1copydf.features_tok)

vmf = TfidfVectorizer(max_df=0.8, min_df=30, tokenizer=lambda i:i, lowercase=False)
UFmf = vmf.fit_transform(finaldata1copydf.features_tok)
UFmf.shape


Xmf = scipy.sparse.csc_matrix(UFmf)

svdmf = TruncatedSVD(n_components=50, n_iter=10, random_state=0)
UmSm = svdmf.fit_transform(Xmf) 
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())
#cumulative = np.cumsum(svdmf.explained_variance_ratio_)
#plt.plot(cumulative, c='blue')
#plt.show()


pdistjaccard = pdist(Xmf.toarray(), 'jaccard')

#lenghtsclusters = [0]
#while(min(lenghtsclusters)<40):
    
k = 40
km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=1)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(UmSm)
cluster_labels = km.fit_predict(UmSm)
print("done in %0.3fs" % (time() - t0))
print()
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_avg = silhouette_score(UmSmconc, cluster_labels)
print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)    

    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    #print("Adjusted Rand-Index: %.3f"
    #      % metrics.adjusted_rand_score(labels, km.labels_))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    #
    #print()
    
    
#origspacecentroids = svdmfconc.inverse_transform(km.cluster_centers_)
#ordercentroids = origspacecentroids.argsort()[:,::-1]
#terms = vmf.get_feature_names()
#for i in range(k):
#    print('cluster %d:'% i,end='')
#    for ind in ordercentroids[i,:5]:
#        print(' %s' % terms[ind],end='')
#    print()
    
    
clusters = km.labels_.tolist()
for count,tl in enumerate(finaldata1notflattened):
    tl['movcluster'] = clusters[count]
    
lenghtsclusters = [len(titlesincluster(ccc)) for ccc in range(k)]
print(min(lenghtsclusters))

plt.plot(lenghtsclusters)



def getgenrehomogeneity(clus):
    genreclus = []
    for i,m in enumerate(finaldata1notflattened):
        if m['movcluster'] == clus:
            genreclus.append(m['genres'])
    countgclus = Counter([g for subl in genreclus for g in subl])
    cglabels, cgvalues = zip(*countgclus.items())
    cgvalues = [c/len(genreclus) for c in cgvalues]
    cgindexes = np.arange(len(cglabels))
    cgwidth = 1
    plt.bar(cgindexes, cgvalues, cgwidth)
    plt.xticks(cgindexes , cglabels, rotation='vertical',
               fontsize='small')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)
    plt.show()   

    return countgclus






titlesincluster(19)[:-20]
def titlesincluster(clus):
    titleindexes = []
    for i,m in enumerate(finaldata1copy):
        if m['movcluster'] == clus:
            titleindexes.append(i)
    titlesincluster = [t for i,t in enumerate(titles) if i in titleindexes]
    return titlesincluster



labels = km.labels_
metrics.silhouette_score(Um, labels, metric='cityblock')

plt.scatter(Um[:, 0], Um[:, 1], c=clusters)



def whereisid(imdbid):
    index = allimdbids.index(imdbid)
    return finaldata1copy[index]['movcluster']

# test imdbids:
#   245429 spirited away
#   133093 matrix
titlesincluster(whereisid(133093))



# make tensor factorization by taking the single attribute matrices
# then tfidf all of them,
# then do the rest
finaldata1notflattened = importMoviesfromJSON('finaldata1_NOTFLATTENED')
finaldata1notflatteneddf = pd.DataFrame({'imdbid': [u['imdbid'] for u in finaldata1notflattened],
            'company': [u['company'] for u in finaldata1notflattened],
            'director': [u['director'] for u in finaldata1notflattened],
            'keywords': [u['keywords'] for u in finaldata1notflattened],
            'genres': [u['genres'] for u in finaldata1notflattened],
            'countries': [u['countries'] for u in finaldata1notflattened],
            'writer': [u['writer'] for u in finaldata1notflattened],
            'languages': [u['languages'] for u in finaldata1notflattened],
            'cast': [u['cast'] for u in finaldata1notflattened],
            'release': [u['release'] for u in finaldata1notflattened],
            'originalmusic': [u['original music'] for u in finaldata1notflattened]
                                         })

    
    
from collections import Counter     
countgenres = Counter([g for subl in list(finaldata1notflatteneddf.genres) for g in subl])
cglabels, cgvalues = zip(*countgenres.items())
cgindexes = np.arange(len(cglabels))
cgwidth = 1
plt.bar(cgindexes, cgvalues, cgwidth)
plt.xticks(cgindexes + cgwidth , cglabels, rotation=90,
           fontsize='small')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.2)
plt.show()   
    

def tfidf(what, mxdf, mndf):
    vmfcom = TfidfVectorizer(max_df=mxdf, min_df=mndf, tokenizer=lambda i:i, lowercase=False)
    COMp = vmfcom.fit_transform(what)
    return vmfcom,COMp



item_matrices = {
            'company': tfidf(finaldata1notflatteneddf.company,0.8,10),
            'director': tfidf(finaldata1notflatteneddf.director,0.8,10),
            'keywords': tfidf(finaldata1notflatteneddf.keywords,0.8,30),
            'genres': tfidf(finaldata1notflatteneddf.genres,0.8,10),
            'countries': tfidf(finaldata1notflatteneddf.countries,0.8,10),
            'writer': tfidf(finaldata1notflatteneddf.writer,0.8,10),
            'languages': tfidf(finaldata1notflatteneddf.languages,0.8,10),
            'cast': tfidf(finaldata1notflatteneddf.cast,0.8,10),
            'release': tfidf(finaldata1notflatteneddf.release,0.8,10),
            'originalmusic': tfidf(finaldata1notflatteneddf.originalmusic,0.8,10)
            }


tryconcatenate = np.concatenate(
                    (scipy.sparse.csc_matrix(item_matrices['company'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['director'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['keywords'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['genres'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['countries'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['writer'][1]).toarray(),
#                     scipy.sparse.csc_matrix(item_matrices['languages'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['cast'][1]).toarray(),
#                     scipy.sparse.csc_matrix(item_matrices['release'][1]).toarray(),
                     scipy.sparse.csc_matrix(item_matrices['originalmusic'][1]).toarray()),
                                 axis=1)

svdmfconc = TruncatedSVD(n_components=200, n_iter=10, random_state=0)
UmSmconc = svdmfconc.fit_transform(tryconcatenate)


distanceUmSmconc = pairwise_distances(UmSmconc,metric='euclidean')
getsimilarityfrommatrix(2488496,10,distanceUmSmconc)



ax = Axes3D(plt.figure())
ax.scatter(UmSmconc[:, 0], UmSmconc[:, 1], UmSmconc[:, 2], alpha=0.6)
plt.show()



dist_outm = pdist(UmSmconc)
Zm = linkage(dist_outm,'ward')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

cm, coph_distsm = cophenet(Zm, dist_outm)
cm
# plot the dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram: average-cityblock')
plt.xlabel('users')
plt.ylabel('distance')
dendrogram(
    Zm,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=6.,  # font size for the x axis labels
    
)
plt.show()











grouped = []
for count, c in enumerate(clusters):
    currg = []
    currg = {'cluster':c, 'tag':kftokenized[count][:20]}
    grouped.append(currg)


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

def gettoptermsforcluster(c,topn):
    import operator
    cluster1kwds = [g['tag'] for g in grouped if g['cluster']==c]
    elemssfreq = list(sorted(createFrquencyTable(cluster1kwds).items(), key=operator.itemgetter(1), reverse=True))
    return elemssfreq[:topn]



gettoptermsforcluster(2,20)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:03:47 2017

@author: michele
"""

import json
import numpy as np
import pandas as pd
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




# work on a copy of the dataset!!!
finaldata1copy = finaldata1.copy()

# make proper preparation of the dataset:
mtitles = gettitles()
titles = []
for m in finaldata1copy:
    titles.append([t['title'] for t in mtitles if t['imdbId']==m['imdbid']][0])

allimdbids = [f['imdbid'] for f in finaldata1copy]

    
featuresnodecade = []
for uf in [u['features'] for u in finaldata1copy]:
    cutkeywords = [f for f in uf if '(keywords)_' in f][:100]
    nodecades = [f for f in uf if '(decade)_' not in f and '(keywords)_' not in f]
    nodecades.extend(cutkeywords)
    
    featuresnodecade.append(nodecades)


finaldata1copydf = pd.DataFrame({'imdbid' : [u['imdbid'] for u in finaldata1copy],
                                 #'numratings' : [imdbratings[imdbratings['imdbId']==imb].rating.count()for imb in allimdbids],
                                 'title' : [t for t in titles],
                                #'features' : [u['string_features'] for u in finaldata1copy],
                                'features_tok' : [u for u in featuresnodecade],
                                #'keywordfeatures' : [k for k in keywordfeatures]
                                })






#------- TF-IDF - find best min and max df
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
vmf = TfidfVectorizer(max_df=0.2, min_df=5, tokenizer=lambda i:i, lowercase=False)
UFmf = vmf.fit_transform(finaldata1copydf.features_tok)
UFmf.shape
Xmf = scipy.sparse.csc_matrix(UFmf)
#[k for k in v2.keys() if k not in v1.keys()]
#v2 = vmf.vocabulary_
#v1 = vmf.vocabulary_

#------- SVD - FIND THE VALUE OF K
#from scipy.sparse.linalg import svds # scipy implementation
#Um, Sm, Vtm = svds(Xmf, 100) #svds(Xmf, min(Xmf.shape)-1)
#
## since scipy returns the matrices in ascending order of singular values,
## they need to be reversed
#Usigma = Um[:,::-1]*(Sm[::-1]) # this is the same output of sklearn


# to inspect the 3D visually
def inspect3d(US,d1,d2,d3):
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure())
    ax.scatter(US[:, d1], US[:, d2], US[:, d3], alpha=0.2)
    plt.show()




from sklearn.decomposition import TruncatedSVD # scikit implementation
# with scikit learn implementation
svdmf = TruncatedSVD(n_components=300, n_iter=10, random_state=0)
UmSm = svdmf.fit_transform(Xmf) # scikit returns the U*S matrix


inspect3d(UmSm,0,1,2)





#------- creation of distance matrix, needed for the hierarchical clustering
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist,cosine,squareform



# COSINE METRIC
dist_outc = pdist(UmSm,'cosine')

# EUCLIDEAN METRIC
dist_oute = pdist(UmSm)







# get similarity 
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


# qualitative comparison of the 3 distance metrics
distanceUFcv = pairwise_distances(UmSm,metric='cosine')
distancecityblock = pairwise_distances(UmSm,metric='cityblock')
distanceeuclid = pairwise_distances(UmSm,metric='euclidean')

cos = getsimilarityfrommatrix(3659388,20,distanceUFcv)
cblock = getsimilarityfrommatrix(2379713,10,distancecityblock)
eucl = getsimilarityfrommatrix(1638355,10,distanceeuclid)
#cosimavgs = simavgs(700)
#plt.plot(cosimavgs)
#def simavgs(n):
#    cosims = []
#    for i in range(n):
#        cosims.append(getsimilarityfrommatrix(1392190,i,distanceUFcv)[1])
#    return cosims


comparisondf = pd.DataFrame({'cosine':cos[0],
                             'cityblock':cblock[0],
                             'euclidean':eucl[0]})


    
import networkx as nx
G = nx.from_numpy_matrix(distanceUFcv) 
nx.draw(G)
pos = nx.random_layout(G) 
    
    
    
    

# ------- HIERARCHICAL CLUSTERING
from scipy.cluster.hierarchy import dendrogram, linkage
Zm = linkage(dist_outc,'average')#, 'average', 'cosine')

from scipy.cluster.hierarchy import cophenet


cm, coph_distsm = cophenet(Zm, dist_outc)
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

hclust(dist_outw,'average')






# plot dendrogram and similarity heatmap.
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

















# ------- K-MEANS

# k-means works on the SVD decomposed matrix, so there is no need to 
# create a distance matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
# perform k-means on k clusters
def kmeans(k,matrix):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=1)
    
    print("Clustering sparse data with %s" % km)
#    t0 = time()
    km.fit(matrix)
    cluster_labels = km.fit_predict(matrix)
#    print("done in %0.3fs" % (time() - t0))
    print()
    from sklearn.metrics import silhouette_samples, silhouette_score
    silhouette_avg = silhouette_score(matrix, cluster_labels)
    print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg)    
    return k,km


# retrieve the representative terms for each cluster
k,km = kmeans(40,UmSm)
origspacecentroids = svdmf.inverse_transform(km.cluster_centers_) # works with scikit
ordercentroids = origspacecentroids.argsort()[:,::-1]
terms = vmf.get_feature_names()
clusterterms = {}    
for i in range(k):
#    clusterterms[i] = ', '.join([terms[ind] for ind in ordercentroids[i,:30]])
    clusterterms[i] = [terms[ind] for ind in ordercentroids[i,:30]]    

for i in range(k):
    print('cluster %d:'% i,end='')
    for ind in ordercentroids[i,:5]: # first 5 terms per cluster
        print(' %s' % terms[ind],end='')
    print()
    


# assign each movie to a cluster
clusters = km.labels_.tolist()
for count,tl in enumerate(finaldata1copy):
    tl['movcluster'] = clusters[count]


# SAVE THE movie CLUSTER LABELING TO NEW JSON
movieclusters = []
for i,m in enumerate(finaldata1copy):
    mc = {'imdbid':     m['imdbid'],
          'cluster':    m['movcluster']}
    movieclusters.append(mc)
    
with open('movieclusterlabels.json', 'w') as fout:
    json.dump(movieclusters, fout)
#movieclusterdicts = {}
#for m in finaldata1copy:
#    movieclusterdicts[m['movcluster']] = m['movcluster']
with open('finaldata1-withclusters.json', 'w') as fout:
    json.dump(finaldata1copy, fout)




# inspect cluster title homogeneity
def titlesincluster(clus): # returns all titles within given cluster
    titleindexes = []
    for i,m in enumerate(finaldata1copy):
        if m['movcluster'] == clus:
            titleindexes.append(i)
    titlesincluster = [t for i,t in enumerate(titles) if i in titleindexes]
    return titlesincluster

# useful function to extract the cluster for a given imdbid
def whereisid(imdbid):
    index = allimdbids.index(imdbid)
    return finaldata1copy[index]['movcluster']


# test imdbids:
#   245429 spirited away
#   133093 matrix
titlesincluster(whereisid(133093))




# inspect cluster size    
lenghtsclusters = [len(titlesincluster(ccc)) for ccc in range(k)]
plt.boxplot(lenghtsclusters)


# inspect genre homogeneity
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





# find the representative movies for each cluster
# representative movies are the most popular (i.e., with more ratings)
import pandas as pd
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
links = pd.read_csv("ml-latest-small/links.csv")

movieratings = pd.merge(ratings, movies, on ='movieId', how='inner')
imdbratings = pd.merge(movieratings, links, on='movieId', how='inner')#.drop(['timestamp',('rating', 'size'), ('rating', 'mean'),  ('rating', 'std'),'title'],1)






# ALTERNATE IMPLEMENTATION WITH TENSOR DECOMPOSITION- 
# BY MEAN OF TENSOR SLICING METHOD
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




# SVD ON LINERIZED TENSOR
svdmfconc = TruncatedSVD(n_components=300, n_iter=10, random_state=0)
UmSmconc = svdmfconc.fit_transform(tryconcatenate)
dist_out_tensor = pairwise_distances(UmSmconc,metric='cosine')
getsimilarityfrommatrix(133093,20,dist_out_tensor)

# inspect3d(UmSmconc,0,1,2)




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



#pth1m = 'ml-1m/'
#movies1m = ['movieId', 'title', 'genres']
#ratings1m = ['userId', 'movieId', 'rating', 'timestamp']
#movies1m = pd.read_table('ml-1m/movies.dat',sep='::',header=None,
#                         names = ['movieId', 'title', 'genres'])
#
#ratings1m = pd.read_table('ml-1m/ratings.dat',sep='::',header=None,
#                         names = ['userId', 'movieId', 'rating', 'timestamp'])
#
#
#m1m = movies1m.movieId
#m100k = movies.movieId
#
#
#mergedmovids = pd.merge(movies1m, movies, on ='movieId', how='inner')




# MOVIE AND USER STATISTICS: MEAN AND STD FOR EACH USERID AND MOVIEID
userstats = ratings.groupby('userId', as_index=False).agg({'rating':[np.size,np.mean,np.std]}) # statistics on users

userstats['rating']['size'].head(50).sort_values(ascending=False).plot(kind="bar",fontsize=6,figsize=(10,5),x="userid")
n = 10
ax = userstats.rating['size'].sort_values(ascending=False).plot(kind="bar",fontsize=6)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
ax.figure.show()
userstats[userstats['rating']['mean']>4]['rating']

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




#analyze mean and std:
us1 = userstats[userstats['rating']['mean']<=3]['rating']
us2 = userstats[userstats['rating']['mean'].between(3, 4, inclusive=False)]['rating']
us3 = userstats[userstats['rating']['mean']>=4]['rating']
#
#fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
#plt.subplot(1,2,1)
#plt.hist(list(us1['mean']), 10, alpha=0.8, label='mean <= 3')
#plt.hist(list(us2['mean']), 10, alpha=0.5, label=' 3 < mean < 4')
#plt.hist(list(us3['mean']), 10, alpha=0.5, label='mean >= 4')
#plt.title("splitted rating mean")
#plt.ylabel('# users')
#plt.xlabel('rating mean')
#plt.legend(loc='upper left')
#plt.show()
#
#plt.subplot(1,2,2)
#plt.hist(list(us1['std']), 10, alpha=0.8, label='mean <= 3')
#plt.hist(list(us2['std']), 10, alpha=0.5, label=' 3 < mean < 4')
#plt.hist(list(us3['std']), 10, alpha=0.5, label='mean >= 4')
#plt.title("standard deviation by splitted rating mean")
#plt.ylabel('# users')
#plt.xlabel('rating mean')
#plt.legend(loc='upper right')
#plt.show()
#      

ratings.boxplot(column='rating')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns

#
import importlib
#importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

metadatadf = metadatadf()
lenkeys = [len(m['keywords'][:100]) for m in metadata]
mkeys = pd.DataFrame({'sizes':lenkeys})


gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
ax1 = plt.subplot(gs[0])
mkeys.plot(kind = 'hist',ax=ax1,bins=50)
plt.title("keywords distribution")
plt.ylabel('# of movies')
plt.xlabel('# of keywords')
plt.show()


ax2 = plt.subplot(gs[1])
mkeys.plot(kind = 'box',ax=ax2)
plt.title("Keyword distribution")
plt.ylabel('counts')
plt.show()


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
                     


plt.subplot(gs[0])#(1,3,1)
vp = sns.violinplot(data=ratings['rating'])
vp.set_xticklabels(['global rating density'])
plt.title("# of ratings per users")
plt.ylabel('# users')
plt.show()

#plt.subplot(1,3,2)
#ax1 = userstats.drop(['userId',('rating','mean'),('rating','std')],1).boxplot()
#plt.title("# of ratings per users")
#plt.ylabel('# users')
#plt.show()

userstats['means'] = userstats['rating']['mean']
userstats['sizes'] = userstats['rating']['size']
result = userstats.sort_values(['sizes'], ascending=False)
ax2 = plt.subplot(gs[1])

userstats.plot.hexbin(x='userId', y='means', gridsize=16,cmap='Blues',ax=ax2)
#userstats.plot.scatter(x='userId', y='sizes',s=userstats['sizes']*.1)#,ax=ax2)

    
plt.title("Mean rating distribution")
plt.ylabel('mean rating')
plt.show()

ax3 = plt.subplot(gs[2])
userstats.means.plot.box(ax=ax3)
plt.title("rating mean per users")
plt.ylabel('rating scale')
plt.show()
#
#plt.subplot(1,4,4)
#userstats.drop(['userId',('rating','size'),('rating','mean')],1).boxplot()
#plt.title("rating standard deviation per users")
#plt.ylabel('standard deviation')
#plt.show()
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
plt.subplot(1,2,1)
sns.distplot(userstats.drop(['userId',('rating','mean'),('rating','std')],1),kde=False)
plt.title("rating long tail distribution")
plt.show()
plt.subplot(1,2,2)
ax1 = userstats.drop(['userId',('rating','mean'),('rating','std')],1).boxplot()
plt.title("# of ratings per users")
plt.ylabel('# users')
plt.show()


sns.lmplot(x='userId',y='means',data=userstats, fit_reg=False,
                size=10, aspect=5,scatter_kws={"s": 15})

 

sns.set_style("darkgrid")
pal = sns.cubehelix_palette(10, start=.5, rot=-.75)#sns.diverging_palette(10, 133, sep=80, n=10)
lm = sns.lmplot(x='userId', y='movieId', hue='rating', data=ratings, fit_reg=False, 
                size=10, aspect=5, palette=pal,scatter_kws={"s": 15})
ax = lm.axes.flat[0]
box = ax.get_position()
ax.set_position([box.x0,box.y0,box.width*0.95,box.height*0.95])
axes = lm.axes
axes[0,0].set_ylim(0,163949) # max movieId is 163949
axes[0,0].set_xlim(0,100) # max userId is 671
lm


sns.violinplot(data=userstats['rating']['size'])
vp = sns.violinplot(data=ratings['rating'])
vp.set_xticklabels(['global rating density'])

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(10,4))

ax1 = plt.subplot(1,3,1)
us1['mean'].plot(kind = 'box',yerr=us1['std'])
#sns.violinplot(data=us1.drop('size',1))
plt.title('mean <= 3, '+str(us1['size'].count())+' users')
plt.show()

plt.subplot(1,3,2,sharey=ax1)
us2['mean'].plot(kind = 'box',yerr=us2['std'])
#sns.violinplot(data=us2.drop('size',1))
plt.title('3 < mean < 4, '+str(us2['size'].count())+' users')
plt.show()

plt.subplot(1,3,3,sharey=ax1)
us3['mean'].plot(kind = 'box',yerr=us3['std'],secondary_y=True)
#sns.violinplot(data=us3.drop('size',1))
plt.title('mean >= 4, '+str(us3['size'].count())+' users')
plt.show()


metadatadfkeys = metadatadf.keywords
mkeys = pd.DataFrame({'sizes':lenkeys})

lenkeys = [len(m['keywords']) for m in metadata]

# import movie metadata from JSON FILE
metadata = []
with open('json_metadataFULL_final', encoding='utf-8') as datafile:
    metadata = json.load(datafile)        
def metadatadf():
    return pd.DataFrame(metadata).fillna(0)        



#==========================================================================================
#                            PREPROCESSING METADATA DATASET
#==========================================================================================


#============================1-FIND ALL N/A-===============================================
# ['company', 'director', 'keywords', 'genres', 'countries',
# 'imdbid', 'writer', 'languages', 'cast', 'release', 'original music']
numberofNA = {}
numberofNA['nancompany'] = 100-len([m['company'] for m in metadata if m['company'] == "N/A"])/len(metadata)
numberofNA['nandirector'] = 100-len([m['director'] for m in metadata if m['director'] == "N/A"])/len(metadata)
numberofNA['nankeywords'] = 100-len([m['keywords'] for m in metadata if m['keywords'] == "N/A"])/len(metadata)
numberofNA['nangenres'] = 100-len([m['genres'] for m in metadata if m['genres'] == "N/A"])/len(metadata)
numberofNA['nancountries'] = 100-len([m['countries'] for m in metadata if m['countries'] == "N/A"])/len(metadata)
numberofNA['nanwriter'] = 100-len([m['writer'] for m in metadata if m['writer'] == "N/A"])/len(metadata)
numberofNA['nanlanguages'] = 100-len([m['languages'] for m in metadata if m['languages'] == "N/A"])/len(metadata)
numberofNA['nancast'] = 100-len([m['cast'] for m in metadata if m['cast'] == "N/A"])/len(metadata)
numberofNA['nanrelease'] = 100-len([m['release'] for m in metadata if m['release'] == "N/A"])/len(metadata)
numberofNA['nanoriginal music'] = 100-len([m['original music'] for m in metadata if m['original music'] == "N/A"])/len(metadata)

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

def inspectmissing():
    def countnan(feature):
        count=len([m[feature] for m in metadata if "N/A" in m[feature]])
        percent = count/len(metadata)
        return percent,count
    
    metadatalabels = list(metadata[1].keys())
    metadatalabels.remove('imdbid')
    counts = []
    percent = []
    for k in metadatalabels:
        p,c = countnan(k)
        counts.append(c)
        percent.append(p)
    
    missingvals = pd.DataFrame({'metadata':metadatalabels,
                                'missing':counts,
                                'relative_missing':percent})
        
    print(missingvals)
    
    
    sns.barplot(x='relative_missing',y='metadata',data=missingvals) 
        #sns.factorplot(x='missing', y='filled', hue='metadata', data=missingvals, kind='bar')    
            
        
























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
        metadata[i]['release'] = [txt.group(0)]
    if(release == 'N/A'):
        metadata[i]['release'] = [[h['title']] for h in moviewo if h['imdbId'] == currimdb][0]
        #print(metadata[i]['release'])


metadatadf = pd.DataFrame(metadata).fillna(0)
inspectmissing()

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


this = [m['keywords'] for m in [hm for hm in metadata if 'Hayao Miyazaki' in hm['director']]]

# keyword analysis
def analyzethis(this):
    keyword_count = analyzefeature(this)
    kwdc = list(list(row) for row in keyword_count)
    analyzefrequency(kwdc)


def analyzelist(this):
#   if a list of words is given as input.
    keyword_count = list(sorted(createFrquencyTable(this).items(), key=operator.itemgetter(1), reverse=True))    
    kwdc = list(list(row) for row in keyword_count)
    analyzefrequency(kwdc)



def agenre(this):
    keyword_count = findkeywordsfor(this)
    kwdc = list(list(row) for row in keyword_count)
    from wordcloud import WordCloud, STOPWORDS
    sns.set_context("poster")
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 14}
    mpl.rc('font', **font)
    #_______________________________________________________
    # I define the dictionary used to produce the wordcloud
    words = dict()
    trunc_occurences = kwdc[0:50]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    tone = 55.0 # define the color of the words
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=300, background_color='white', 
                          max_words=1628,relative_scaling=1,
    #                      color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    return wordcloud
    ax1.imshow(wordcloud, interpolation="bilinear")
#    plt.title("Keyword popularity")
    ax1.axis('off')


def analyzefrequency(kwdc):
    from wordcloud import WordCloud, STOPWORDS
    sns.set_context("poster")
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 14}
    mpl.rc('font', **font)
    #_____________________________________________
    # Function that control the color of the words
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING: the scope of variables is used to get the value of the "tone" variable
    # I could not find the way to pass it as a parameter of "random_color_func()"
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #def random_color_func(word=None, font_size=None, position=None,
    #                      orientation=None, font_path=None, random_state=None):
    #    h = int(360.0 * tone / 255.0)
    #    s = int(100.0 * 255.0 / 255.0)
    #    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    #    return "hsl({}, {}%, {}%)".format(h, s, l)
    #_____________________________________________
    # UPPER PANEL: WORDCLOUD
    fig = plt.figure(1, figsize=(18,24))
    ax1 = fig.add_subplot(2,1,1)
    #_______________________________________________________
    # I define the dictionary used to produce the wordcloud
    words = dict()
    trunc_occurences = kwdc[0:50]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    tone = 55.0 # define the color of the words
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=300, background_color='white', 
                          max_words=1628,relative_scaling=1,
    #                      color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    plt.title("Keyword popularity")
    ax1.axis('off')
    #_____________________________________________
    # LOWER PANEL: HISTOGRAMS
    ax2 = fig.add_subplot(2,1,2)
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k,i in enumerate(trunc_occurences)]
    x_label = [i[0] for i in trunc_occurences]
    plt.xticks(rotation=35, ha='right', rotation_mode='anchor', fontsize = 7)
    plt.yticks(fontsize = 15)
    plt.xticks(x_axis, x_label)
    plt.ylabel("occurences", fontsize = 10, labelpad = 10)
    ax2.bar(x_axis, y_axis, align = 'center', color='g')
    #_______________________

    plt.show()










def awcplot(f1):
    # uncomment 
    keyword_count = analyzefeature(f1)
    kwdc = list(list(row) for row in keyword_count)
    
    from wordcloud import WordCloud, STOPWORDS
    sns.set_context("poster")
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 14}
    mpl.rc('font', **font)
    #_______________________________________________________
    # I define the dictionary used to produce the wordcloud
    words = dict()
    trunc_occurences = kwdc[0:50]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    tone = 55.0 # define the color of the words
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=300, background_color='white', 
                          max_words=1628,relative_scaling=1,
    #                      color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    return wordcloud
    ax1.imshow(wordcloud, interpolation="bilinear")
#    plt.title("Keyword popularity")
    ax1.axis('off')
    
    
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,4))
fig = plt.figure()


f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(awcplot('director'), interpolation="bilinear")
axarr[0,0].autoscale(False)
axarr[0,0].set_title('Directors',
             bbox={'facecolor':'k', 'pad':1},color='w', fontsize=16);
axarr[0,0].set_adjustable('box-forced')
axarr[0,0].axis('off')


axarr[1,0].imshow(awcplot('cast'), interpolation="bilinear")
axarr[1,0].autoscale(False)
axarr[1,0].set_title('Cast',
             bbox={'facecolor':'k', 'pad':1},color='w', fontsize=16);
axarr[1,0].set_adjustable('box-forced')
axarr[1,0].axis('off')

axarr[0,1].imshow(awcplot('writer'), interpolation="bilinear")
axarr[0,1].autoscale(False)
axarr[0,1].set_title('Writers',
             bbox={'facecolor':'k', 'pad':1},color='w', fontsize=16);
axarr[0,1].set_adjustable('box-forced')
axarr[0,1].axis('off')


axarr[1,1].imshow(awcplot('original music'), interpolation="bilinear")
axarr[1,1].autoscale(False)
axarr[1,1].set_title('Composers',
             bbox={'facecolor':'k', 'pad':1},color='w', fontsize=16);
axarr[1,1].set_adjustable('box-forced')
axarr[1,1].axis('off')



def findkeywordsfor(genre):
    import operator
    wordlist = []
    for m in metadata:
        # print the first language for a movie with country = country
        if(genre in m['genres'] and 'N/A' not in m['keywords']):
            wordlist.append(m['keywords'][:10])
    keyfreq = createFrquencyTable(wordlist)
    commonkey = list(sorted(keyfreq.items(), key=operator.itemgetter(1), reverse=True)[10:30])
    return commonkey
# genre closer inspection
f, axarr = plt.subplots(3,3)
genres = [g[0] for g in analyzefeature('genres')]

idx = 0
for i in range(3):
    for j in range(3):
        axarr[i,j].imshow(agenre(genres[idx]), interpolation="bilinear")
        axarr[i,j].autoscale(False)
        axarr[i,j].set_title(genres[idx],
                     bbox={'facecolor':'k', 'pad':1},color='w', fontsize=14);
        axarr[i,j].set_adjustable('box-forced')
        axarr[i,j].axis('off')
        idx += 1




def findproficiency(this):
    keyword_count = analyzefeature(this)
    kwdc = list(list(row) for row in keyword_count)
    counts = [k[1] for k in kwdc]
    return pd.DataFrame({'movies':counts})
fig, ax = plt.subplots(nrows=1,ncols=5,figsize=(10,4))

ax1 = plt.subplot(1,5,1)
findproficiency('director')['movies'].plot(kind = 'box')
#sns.violinplot(data=us1.drop('size',1))
plt.title('movies by director')
plt.show()

plt.subplot(1,5,2)
findproficiency('cast')['movies'].plot(kind = 'box')
#sns.violinplot(data=us1.drop('size',1))
plt.title('movies by cast')
plt.show()

plt.subplot(1,5,3)
findproficiency('writer')['movies'].plot(kind = 'box')
#sns.violinplot(data=us1.drop('size',1))
plt.title('movies by writer')
plt.show()

plt.subplot(1,5,4)
findproficiency('original music').boxplot()
#sns.violinplot(data=us1.drop('size',1))
plt.title('movies by composer')
plt.show()

plt.subplot(1,5,5)
findproficiency('company')['movies'].plot(kind = 'hist')
#sns.violinplot(data=us1.drop('size',1))
plt.title('movies by production company')
plt.show()








# decade analysis
releases = list(metadatadf.release)
df_initial = pd.DataFrame({'title_year':[int(k[0]) for k in releases]})
df_initial['decade'] = df_initial['title_year'].apply(lambda x:int(x/10)*10 -1900)

def get_stats(gr):
    return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}
#______________________________________________________________
# Creation of a dataframe with statitical infos on each decade:
test = df_initial['title_year'].groupby(df_initial['decade']).apply(get_stats).unstack()
sns.set_context("poster", font_scale=0.85)
#_______________________________
# funtion used to set the labels
def label2(s):
    val = (1900 + s, s)[s < 100]
    chaine = '' if s < 50 else "{}'s".format(int(val))
    return chaine
#____________________________________
plt.rc('font', weight='normal',size=10)

labels = [label2(s) for s in  test.index]
sizes  = test['count'].values
explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(12)]

cm = plt.get_cmap('tab20')
cs1=cm(np.linspace(0, 1.0, len(labels)))


plt.pie(sizes, explode = explode, labels=labels, colors=cs1,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0,
       pctdistance=1.1, labeldistance=1.2)
plt.axis('equal')
#plt.legend(pie[0], labels, loc="upper left")
plt.set_title('% of films per decade',
             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
df_initial.drop('decade', axis=1, inplace = True)

#import plotly
#from plotly.graph_objs import Scatter, Layout
#plotly.offline.plot({
#"data": [
#    go.Pie(labels=klab, values=kvals,
#           pct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
#           textposition='outside',textinfo='label')
#],
#"layout": Layout(
#    title="hello world"
#)
#})


def label(kwdc):
    kwdc2 = []
    kminor = 0
    for kv in kwdc:
        if(kv[1]<90):
            kminor += kv[1] # add the value to minor labels
        else:
            kwdc2.append(kv)
    kwdc2.append(['Others',kminor])
    return kwdc2


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
plt.rc('font', size=8)
plt.rc('xtick', labelsize=8)
plt.rc('font', weight='normal')
from matplotlib import cm


# --------countries
keyword_count = analyzefeature('countries')
kwdc = list(list(row) for row in keyword_count)
kwdc = label(kwdc)
klab = [k[0] for k in kwdc]
kvals = [k[1] for k in kwdc]

cm = plt.get_cmap('tab20')
cs1=cm(np.linspace(0, 1.0, len(kwdc)))


ax1 = plt.subplot(1,2,1)
plt.axis("equal")
pie = plt.pie(kvals, labels=klab,
        explode=[0.2 if kvals[i] < 90 else 0.01 for i in range(len(kwdc))],
        shadow=False, startangle=0,colors=cs1,
        autopct= lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
        pctdistance=1.1, labeldistance=1.2)#.set_fontsize(4)
#plt.legend(pie[0], klab, loc="upper left")
plt.show()

# --------languages
keyword_countl = analyzefeature('languages')
kwdcl = list(list(row) for row in keyword_countl)
kwdcl = label(kwdcl)
klabl = [k[0] for k in kwdcl]
kvalsl = [k[1] for k in kwdcl]


cs2=cm(np.linspace(0, 1.0, len(kwdcl)))
plt.subplot(1,2,2)
plt.axis("equal")
pie = plt.pie(kvalsl, labels=klabl,
        explode=[0.2 if kvalsl[i] < 90 else 0.01 for i in range(len(kwdcl))],
        shadow=False, startangle=0,colors=cs2,
        autopct= lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
        pctdistance=1.1, labeldistance=1.2)
#plt.legend(pie[0], klab, loc="upper left")
plt.show()









# from 1 to 10 
#castnr = []
#for m in metadata:
#    castnr.append(len(m['cast']))
#plt.hist(castnr) # distribution of cast, no reason to limit this.   

keynr = []
for m in metadata:
    keynr.append(len(m['keywords']))
plt.hist(keynr,bins=100) # distribution of keywords, we need to limit them  
plt.title("Keyword distribution")
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
    for m in metadata:
        # print the first language for a movie with country = country
        if('N/A' not in m['languages'] and country in m['countries']):
            wordlist.append(m['languages'])
    countrylangfreq = createFrquencyTable(wordlist)
    commonlang = list(sorted(countrylangfreq.items(), key=operator.itemgetter(1), reverse=True)[0])[0]
    return commonlang


#replace the N/A of language with the most common language for that country
for i,m in enumerate(metadata):
    if('N/A' in m['languages'] and 'N/A' not in m['countries']):
        print(m['imdbid'])
        print(m['languages'])
        print(m['countries'])
        print(findlanguagefor(m['countries'][0]))
        metadata[i]['languages'] = findlanguagefor(m['countries'][0])
        
        
# do the same for the country
        # get the countries for a language
def findcountryfor(language):
    import operator
    wordlist = []
    for m in metadata:
        # print the first language for a movie with country = country
        if(language in m['languages'] and 'N/A' not in m['countries']):
            wordlist.append(m['countries'])
    countrylangfreq = createFrquencyTable(wordlist)
    commoncountry = list(sorted(countrylangfreq.items(), key=operator.itemgetter(1), reverse=True)[0])[0]
    return commoncountry
        
# replace
for i,m in enumerate(metadata):
    if('N/A' in m['countries'] and 'N/A'  in m['languages']):
        print(m['imdbid'])
        print(m['languages'])
        print(m['countries'])
        metadata[i]['countries'] = ['USA']
        metadata[i]['languages'] = ['English']



# companies
imdbnocompanies = [m['imdbid'] for m in metadata if 'N/A' in m['company']]





#================ CLEAN KEYWORDS



keygenres = [m['imdbid'] for m in metadata if 'thriller' in [x.lower() for x in m['keywords']]]


# find the most occurring keyword(s) for a particular genre
def findkeywordsfor(genre):
    import operator
    wordlist = []
    for m in metadata:
        # print the first language for a movie with country = country
        if(genre in m['genres'] and 'N/A' not in m['keywords']):
            wordlist.append(m['keywords'][:10])
    keyfreq = createFrquencyTable(wordlist)
    commonkey = list(sorted(keyfreq.items(), key=operator.itemgetter(1), reverse=True)[:15])
    commonkey = [c[0] for c in commonkey]
    return commonkey
        
# replace
for i,m in enumerate(metadata):
    if('N/A' in m['keywords'] and 'N/A' not in m['genres']):
        print(m['imdbid'])
        print(m['languages'])
        print(m['countries'])
        metadata[i]['keywords'] = findkeywordsfor(m['genres'][0])


#================ CLEAN KEYWORDS EEEEND!!!



#================ CLEAN directors












#============================0-ANALYZE FREQUENCIES-=====================================
def analyzefeature(feature):
    import operator
    elems = []
#    for m in metadataklim:
    for m in metadata:
        if('N/A' not in m[feature]):
            elems.append(m[feature])
#        elif('N/A' in m[feature]):
#            elems.append(['N/A'])
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
    # wo release
# keep the movies with N/A:
    # cast
    # director
    # company
    # original music
    # writer

metadatacleaned = []
for i,m in enumerate(metadata):
    if('N/A' not in m['countries'] and 'N/A' not in m['keywords'] and 
       'N/A' not in m['release']):
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
addtruncatedrelease(metadatacleaned) # add decade to the movie metadata!!!
metadatawoogw = importMoviesfromJSON('METADATANOMUSICANDWRITER.json')
metadatanomissing = importMoviesfromJSON('METADATANOMISSING.json')
addtruncatedrelease(metadatanomissing)

f1 = getfeaturedataset(metadatacleaned[0:10])
[f['imdbid'] for f in f1[0:10]]

def addtruncatedrelease(metadatacleaned):
    for i,m in enumerate(metadatacleaned):
        metadatacleaned[i]['decade'] = [str(int(int(m['release'][0])/10)*10)]
#
#def truncatekeywords(meta):
#    for i,m in enumerate(meta):
#        meta[i]['keywords'] = m['keywords'][:50]

#truncatekeywords(metadatacleaned)

def getlistoffeatures(d): 
    # d is the dictionary from the metadata files of a single movie
    # need to link the feature to its type (director, company, cast, etc.)
    listoffeatures = []
    
    for k,v in d.items():
        if(type(v)!=int):
#            print(d['imdbid'])
            for item in v:
                listoffeatures.append('('+k+')'+'_'+'('+item+')')
        elif(type(v)==str):
            listoffeatures.append('('+k+')'+'_'+'('+v+')')
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


[type(v) for v in metadatacleaned[1].values()]

getandsavedataset(metadatacleaned,'finaldata1') # dataset 1 movies removed for errors in imdb features but N/A left for other features
getandsavedataset(metadatawoogw,'finaldata2') # dataset 3 as above but without OST and writer (the features with most missing vals in movies)
getandsavedataset(metadatanomissing,'finaldata3') # dataset 3 no missing features




# make a get feature dataset simple without including the type of metadata (genre, cast, etc.)
def getlistoffeaturessimple(d): 
    # d is the dictionary from the metadata files of a single movie
    # need to link the feature to its type (director, company, cast, etc.)
    listoffeatures = []
    for k,v in d.items():
        if(type(v)!=int):
            for item in v:
                listoffeatures.append('('+item+')')
    return listoffeatures # returns the imdbid and the list of features

def getfeaturedatasetsimple(data): 
    # for all movies in the dataset data get the features and the corresp imdbid
    finaldata = []
    for m in data:
        curr = {'imdbid' :   m['imdbid'],
                'features' : getlistoffeatures(m),
                'simplefeatures' : getlistoffeaturessimple(m)}
        finaldata.append(curr)
    return finaldata

def getandsavedatasetsimple(metadatafile,jsonname):
    finaldata = getfeaturedatasetsimple(metadatafile)
    with open(jsonname, 'w') as fout:
        json.dump(finaldata, fout)

getandsavedatasetsimple(metadatacleaned,'simplefinaldata1')





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
imdbratings = pd.merge(movieratings, links, on='movieId', how='inner')#.drop(['timestamp',('rating', 'size'), ('rating', 'mean'),  ('rating', 'std'),'title'],1)
imdbratings['binrating'] = np.where(imdbratings['normrating']>0, 1, -1) # binarized ratings

selectedratings = imdbratings[imdbratings['binrating']==1]
selectedratings = imdbratings
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
                userXratings[i]['moviesratings'][j]['features'] = mf['features']#[m for m in mf['features'] if '(release)_' not in m]
             #       

# the dataset of user features!!!!
userXfeatures = []
for i,uid in enumerate(userXratings):
    features = [ff['features'] for ff in userXratings[i]['moviesratings'] if('features' in ff.keys())]
    uxf = {'userid' : str(uid['userid']),
           'features' : [item for sublist in features for item in sublist]}
    userXfeatures.append(uxf)
    



# save the user features dataset to JSON!!
with open('userXfeatures1-fullratings', 'w') as fout:
    json.dump(userXfeatures, fout)
    







## no NO NOT ANYMORE
## do the same with simple features:
#simpledataset1 = getdataset('simplefinaldata1')
#
## make a merge with the dataset features!!!
#for i,ur in enumerate(userXratings):
#    curru = ur['userid']
#    for j,mr in enumerate(ur['moviesratings']):
#        for mf in simpledataset1:            
#            if(mr['imdbId'] == mf['imdbid']):
#                userXratings[i]['moviesratings'][j]['features'] = mf['simplefeatures']
#
#
## the dataset of user features!!!!
#userXfeatures = []
#for i,uid in enumerate(userXratings):
#    features = [ff['features'] for ff in userXratings[i]['moviesratings'] if('features' in ff.keys())]
#    uxf = {'userid' : str(uid['userid']),
#           'features' : [item for sublist in features for item in sublist]}
#    userXfeatures.append(uxf)
#
## save the user features dataset to JSON!!
#with open('simpleuserXfeatures1', 'w') as fout:
#    json.dump(userXfeatures, fout)
#    
#









    
# ========= NOW WE HAVE THE USERXFEATURE DATASET with the FINALDATA1 movie dataset
userXfeatures = importMoviesfromJSON('userXfeatures1')
# transform a little bit the features in the dataset
for i,uf in enumerate(userXfeatures):
    stringfeature = uf['features']
    strf = ''
    for s in stringfeature:
        strf += ' '+s
    userXfeatures[i]['string_features'] = strf


# make a dataframe with all features and userids
userXfeaturesdf = pd.DataFrame({'userid' : [u['userid'] for u in userXfeatures],
                                'features' : [u['string_features'] for u in userXfeatures]})





# perform TFIDF on the dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_df=0.8, min_df=2)
UF = v.fit_transform(userXfeaturesdf['features'])

# svd



# perform SVD on the TFIDF transformed matrix (UF)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=670, n_iter=10, random_state=0)
svd.fit(UF) 
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())

cumulative = np.cumsum(svd.explained_variance_ratio_)
plt.plot(cumulative, c='blue')
plt.show()


from scipy.sparse.linalg import svds
import scipy

X = scipy.sparse.csc_matrix(UF)
U, S, Vt = svds(X, 400)
cumulative = np.cumsum(np.power(sorted(S, reverse=True),2)/sum(np.power(sorted(S, reverse=True),2)))
plt.plot(cumulative, c='blue')
plt.show() # at around 400 singular values the shit is high



# t-sne visualization
from bhtsne import tsne
tsne_U = tsne(U)
# t-sne plot
plt.scatter(tsne_U[:, 0], tsne_U[:, 1], alpha=0.1)

# svd 670 plot
plt.scatter(U[:, 0], U[:, 1], alpha=0.1)



from sklearn.manifold import TSNE
tsne__U = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000)
tsne_results = tsne__U.fit_transform(U)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.1)
#chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
#        + geom_point(size=70,alpha=0.1) \
#        + ggtitle("tSNE dimensions colored by digit")
#chart



tsne2_U = TSNE(n_components=2,random_state=0)
tsne2_results = tsne2_U.fit_transform(U)
plt.figure()
plt.scatter(tsne2_results[:, 0], tsne2_results[:, 1])
plt.show()










# to do further processing on matlab
def savematrix(arr,path):
    import numpy, scipy.io    
    scipy.io.savemat(path, mdict={'arr': arr})

savematrix(X,'matrixusertfidfdataset1.mat')

































# try 2 with simple dataset features without key
# ========= NOW WE HAVE THE USERXFEATURE DATASET with the FINALDATA1 movie dataset
userXfeatures = importMoviesfromJSON('simpleuserXfeatures1')
# transform a little bit the features in the dataset
for i,uf in enumerate(userXfeatures):
    stringfeature = uf['features']
    strf = ''
    for s in stringfeature:
        strf += ' '+s
    userXfeatures[i]['string_features'] = strf


# make a dataframe with all features and userids
userXfeaturesdf = pd.DataFrame({'userid' : [u['userid'] for u in userXfeatures],
                                'features' : [u['string_features'] for u in userXfeatures]})





# perform TFIDF on the dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_df=0.5, min_df=2)
UF = v.fit_transform(userXfeaturesdf['features'])


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=670, n_iter=10, random_state=0)
svd.fit(UF) 
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())

cumulative = np.cumsum(svd.explained_variance_ratio_)
plt.plot(cumulative, c='blue')
plt.show()


from scipy.sparse.linalg import svds
import scipy

X = scipy.sparse.csc_matrix(UF)
U, S, Vt = svds(X, 670)
cumulative = np.cumsum(np.power(sorted(S,reverse=True),2)/sum(np.power(sorted(S,reverse=True),2)))
plt.plot(cumulative, c='blue')
plt.show()


# t-sne visualization
def tsnevis(U):
    from bhtsne import tsne
    tsne_U = tsne(U)
    # t-sne plot
    plt.scatter(tsne_U[:, 0], tsne_U[:, 1], alpha=0.1)
    plt.show()

tsnevis(U)
# svd 670 plot
plt.scatter(U[:, 0], U[:, 1], alpha=0.1)














#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:55:58 2017

extract the metadata from imdb and create a JSON
for each imdbid in the 100k movielens dataset

interesting metadata are:
    director
    writer
    composer
    cast
    genres
    plot keywords
    countries
    languages
    releases
    companies

@author: michele
"""

import json
from imdb import IMDb
import pandas as pd

# read the links csv and extract the imdbids
movies = pd.read_csv('ml-latest-small/links.csv')
imdbids = movies.imdbId.tolist()

# initialise imdb access
imdb = IMDb()
# test imdbid nausicaa 0087544
#nausicaa = imdb.get_movie('0245429')
#imdb.update(nausicaa, 'keywords')
#print nausicaa['keywords']

# create data structure to save the metadata
metadata = []
metadata4000 = []

for i,m in enumerate(imdbids[0:4000]):
    curr = imdb.get_movie(m)
    imdb.update(curr, 'keywords') # required in order to get ALL keywords
    imdb.update(curr, 'release dates') # required in order to get ALL release dates
    try:
        music = [a['name'] for a in curr['original music']]
    except KeyError:
        music = 'N/A' # if movie does not have a composer
    
    try:
        writer = [a['name'] for a in curr['writer']]
    except KeyError:
        writer = 'N/A' # movie w/o writer

    try:
        cast = [a['name'] for a in curr['cast']]
    except KeyError:
        cast = 'N/A' # just in case
        
    try:
        director = [a['name'] for a in curr['director']]
    except KeyError:
        director = 'N/A' # movie w/o director
        
    try:
        countries = curr['countries']
    except KeyError:
        countries = 'N/A' # movie w/o countries
        
    try:
        genres = curr['genres']
    except KeyError:
        genres = 'N/A' # movie w/o genres
        
    try:
        keywords = curr['keywords']
    except KeyError:
        keywords = 'N/A' # movie w/o keywords
        
    try:
        languages = curr['languages']
    except KeyError:
        languages = 'N/A' # movie w/o languages
        
    try:
        company = [a['name'] for a in curr['production company']]
    except KeyError:
        company = 'N/A' # movie w/o production company
        
    try:
        release = curr['release dates'][0] # only the first is interesting, it refers to the release in the production country
    except KeyError:
        release = 'N/A' # movie w/o release date (just in case it's not in imdb)
    
    
    
    currm = {'imdbid' : m,
             'director' : director,
             'writer' : writer,
             'genres' : genres,
             'cast' : cast,
             'keywords' : keywords,
             'countries' : countries,
             'languages' : languages,
             'original music' : music,
             'company' : company,
             'release' : release
             }
    
    print str(i)+' id: '+str(m)
    metadata.append(currm)
    metadata4000.append(currm)
    
with open('json_metadata', 'a') as f:
    json.dump(metadata, f)

    
    
    
    
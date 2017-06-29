#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:24:56 2017

METADATA SCRAPER FROM IMDB

@author: michele
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

#all_ratings = pd.read_csv("ml-20m/ratings.csv")
# all_ratings.head()
#all_movies = pd.read_csv("ml-20m/movies.csv")

all_links = pd.read_csv("ml-20m/links.csv")

# get all imdbids and movieids
all_imdbids = all_links.imdbId
all_movieids = all_links.movieId
# all movie metadata go here
# prepare the dictionary where to list the metadata
movie_metadata = {'sdf':[f for f in all_imdbids],
                  'sdf2':[f for f in all_movieids]}


# initially with the IMDBID
#for m in all_imdbids:
#    currmetadata = {'imdbid':m,
#                    'movielensid':}
#    movie_metadata.append(currmetadata)

# get all urls from the imdbids
    #============================== 1) EXTRACT EVERYTHING FROM IMDB
imdbURL = "http://www.imdb.com/title/tt"

for f in all_imdbids[0:10]:
    req = requests.get(imdbURL + str(f))
    soup = BeautifulSoup(req.content, 'lxml')
    for movie in soup.findAll('td','title'):
        title = movie.find('a').contents[0]
        genres = movie.find('span','genre').findAll('a')
        genres = [g.contents[0] for g in genres]
        runtime = movie.find('span','runtime').contents[0]
        rating = movie.find('span','value').contents[0]
        year = movie.find('span','year_type').contents[0]
        imdbID = movie.find('span','rating-cancel').a['href'].split('/')[2]
        print(title, genres,runtime, rating, year)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:24:49 2017

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
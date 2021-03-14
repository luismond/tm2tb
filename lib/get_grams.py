#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:08:49 2021
@author: user
ngram generator, ignore stopwords at edges
"""

def get_bigrams(s, stops):
       g = list(zip(s,s[1:]))
       g = [' '.join((i[0],i[1])) for i in g 
             if i[0].isalpha() and not i[0].lower() in stops 
             and i[1].isalpha() and not i[1].lower() in stops]
       if len(g)>0:
           return g
       else:
           return []
       
def get_trigrams(s, stops):
    g = list(zip(s,s[1:],s[2:]))
    g = [' '.join((i[0],i[1],i[2])) for i in g 
          if i[0].isalpha() and not i[0].lower() in stops 
          and i[2].isalpha() and not i[2].lower() in stops]
    if len(g)>0:
        return g
    else:
        return []

def get_fourgrams(s, stops):
    g = list(zip(s,s[1:],s[2:],s[3:]))
    g = [' '.join((i[0],i[1],i[2],i[3])) for i in g 
          if i[0].isalpha() and not i[0].lower() in stops 
          and i[3].isalpha() and not i[3].lower() in stops]
    if len(g)>0:
        return g
    else:
        return []

def get_fivegrams(s, stops):
    g = list(zip(s,s[1:],s[2:],s[3:],s[4:]))
    g = [' '.join((i[0],i[1],i[2],i[3],i[4])) for i in g 
          if i[0].isalpha() and not i[0].lower() in stops 
          and i[3].isalpha() and not i[4].lower() in stops]
    if len(g)>0:
        return g
    else:
        return []

def get_grams(s, stops):
    b = get_bigrams(s, stops)
    t = get_trigrams(s, stops)
    f = get_fourgrams(s, stops)
    v = get_fivegrams(s, stops)
    gs = b+t+s+f+v
    new_gs = []
    for g in gs:
        if all([len(i)<2 for i in g]) == True:
            new_gs.append(g)
    return new_gs
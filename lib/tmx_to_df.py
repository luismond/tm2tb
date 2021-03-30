#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 01:43:37 2021

@author: user
"""
import pandas as pd
import xmltodict
from collections import OrderedDict

# tmx to df 
def tmx_to_df(filepath):
    with open('{}'.format(filepath)) as fd:
        doc = xmltodict.parse(fd.read())
      
    sources = [t['tuv'][0]['seg'] for t in doc['tmx']['body']['tu']]
    targets = [t['tuv'][1]['seg'] for t in doc['tmx']['body']['tu']]
    
    def get_tmx_text(unit):
        try:
            if type(unit) == str:
                return unit
            if type(unit) == OrderedDict:
                return unit['#text']
        except:
            return ''
        
    sources = [get_tmx_text(unit) for unit in sources]
    targets = [get_tmx_text(unit) for unit in targets]
    
    src_trg = list(zip(sources,targets))    
    df = pd.DataFrame(src_trg)
    df.columns = ['src', 'trg']
    df = df[df['src'].str.len()>0]
    df = df[df['trg'].str.len()>0]
    
    return df
    

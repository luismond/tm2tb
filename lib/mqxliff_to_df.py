#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 01:43:37 2021
@author: user
"""
import pandas as pd
import xmltodict

# mqxliff to csv (memoQ)
def mqxliff_to_df(filepath):
    with open('{}'.format(filepath), encoding='utf8') as fd:
       doc = xmltodict.parse(fd.read())
    file = doc['xliff']['file']
    units = file['body']['trans-unit']
    sources = [unit['source'] for unit in units]
    targets = [unit['target'] for unit in units]
    #src_lang = doc['xliff']['file']['@source-language'].split('-')[0]
    #trg_lang = doc['xliff']['file']['@target-language'].split('-')[0]
    src_trg = list(zip(sources,targets))    
    #get segments if both languages have a #text node
    src_trg = [i for i in src_trg if '#text' in i[0] and '#text' in i[1]] 
    #get text nodes from segs
    src_trg = [(i[0]['#text'],i[1]['#text']) for i in src_trg]
    df = pd.DataFrame(src_trg)
    df.columns = ['src', 'trg']
    return df
        
# mqxliff to csv (memoQ)
def mqxliff_to_src(filepath):
    with open('{}'.format(filepath), encoding='utf8') as fd:
       doc = xmltodict.parse(fd.read())
    
    file = doc['xliff']['file']
    units = file['body']['trans-unit']
    sources = [unit['source'] for unit in units]
    #targets = [unit['target'] for unit in units]
    #src_lang = doc['xliff']['file']['@source-language'].split('-')[0]
    #trg_lang = doc['xliff']['file']['@target-language'].split('-')[0]
    #src_trg = list(zip(sources,targets))    
    #get segments if both languages have a #text node
    sources = [u['#text'] for u in sources]
    #src_trg = [i for i in src_trg if '#text' in i[0] and '#text' in i[1]] 
    #get text nodes from segs
    #src_trg = [(i[0]['#text'],i[1]['#text']) for i in src_trg]
    df = pd.DataFrame(sources)
    df.columns = ['src']
    return df
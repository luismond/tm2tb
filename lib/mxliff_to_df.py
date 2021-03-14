#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:07:57 2021

@author: user
"""
import pandas as pd
import xmltodict

# mxliff to df (memsource)
def mxliff_to_df(filepath):
    with open('{}'.format(filepath), encoding='utf8') as fd:
        doc = xmltodict.parse(fd.read())
    file = doc['xliff']['file']
    #get translation units
    units = [unit['trans-unit'] for unit in file['body']['group']]
    # get source and target segments
    sources = [unit['source'] for unit in units]
    targets = [unit['target'] for unit in units]
    # get language ids
    src_lang = file['@source-language']
    trg_lang = file['@target-language']
    # zip segments
    src_trg = list(zip(sources,targets)) 
    # make dataframe
    df = pd.DataFrame(src_trg)
    # fill empty segments
    df = df.fillna('')
    # name columns
    df.columns = ['src', 'trg']
    print('mxliff to df done')
    return df
     
# mxliff to df (memsource)
def mxliff_to_src(filepath):
    with open('{}'.format(filepath), encoding='utf8') as fd:
        doc = xmltodict.parse(fd.read())
    file = doc['xliff']['file']
    #get translation units
    units = [unit['trans-unit'] for unit in file['body']['group']]
    # get source and target segments
    sources = [unit['source'] for unit in units]
    #targets = [unit['target'] for unit in units]
    # get language ids
    #src_lang = file['@source-language']
    #trg_lang = file['@target-language']
    # zip segments
    #src_trg = list(zip(sources,targets)) 
    # make dataframe
    df = pd.DataFrame(sources)
    # fill empty segments
    #df = df.fillna('')
    # name columns
    df.columns = ['src']
    print('mxliff to src done')
    return df
     

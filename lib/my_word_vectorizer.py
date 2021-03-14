#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:31:24 2021

@author: user

vectorize preprocessed tm, return tm and gensim objects
"""

# VEC
from gensim import corpora
from gensim.models import TfidfModel
def my_word_vectorizer(tm):
# GET SRC AND TRG DICTIONARIES
    src_dict = corpora.Dictionary(tm['src'].tolist())
    trg_dict = corpora.Dictionary(tm['trg'].tolist())
    # GET BOW
    tm['srcb'] = tm['src'].apply(lambda l: src_dict.doc2bow(l))
    tm['trgb'] = tm['trg'].apply(lambda l: trg_dict.doc2bow(l))
    #GET TFIDF MODELS
    src_tfidf_model = TfidfModel(corpus=tm['srcb'].tolist(), smartirs='bpc')
    trg_tfidf_model = TfidfModel(corpus=tm['trgb'].tolist(), smartirs='bpc')
    src_tfidf_model_d = dict(src_tfidf_model[list(src_dict.dfs.items())])
    trg_tfidf_model_d = dict(trg_tfidf_model[list(trg_dict.dfs.items())])
    tm = tm.drop(columns=['srcb','trgb'])
    return (tm, src_dict, trg_dict, src_tfidf_model_d, trg_tfidf_model_d)

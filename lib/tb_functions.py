#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:48:55 2021
@author: user
get tb fq, tfidf stats, drop outliers
"""
def tb_prune_fqs(tb):
     tb['fq_diff_ratio'] = [get_diff_ratio(i)
                            for i in list(zip(tb['src_fq'], tb['trg_fq']))]
     tb['tfidf_diff_ratio'] = [get_diff_ratio(i)
                               for i in list(zip(tb['src_tfidf'], tb['trg_tfidf']))]
     tb = tb[tb['fq_diff_ratio'] < 1.8]
     tb = tb[tb['tfidf_diff_ratio'] < 1.8]
     tb = tb.sort_values(by='pair_fq',ascending=False)
     return tb
 
def get_diff_ratio(i):
    if i[0] == i[1]:
        r = i[0]/i[1]
    if i[0] > i[1]:
        r = i[0]/i[1]
    if i[0] < i[1]:
        r = i[1]/i[0]
    return r

   
def get_mt_match(t):
    mt_cands = t[0]
    trg = t[1]
    if trg.lower() in [e.lower() for e in mt_cands]:
        return trg
    

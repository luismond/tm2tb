#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 05:13:48 2021

@author: user
"""
from lib.get_azure_dict_lookup import get_azure_dict_lookup
from lib.get_azure_translation import get_azure_translation

def tb_to_azure(tb, src_det, trgdet):
    #process TB candidates, use Azure to perform dictionary lookup and MT
    # LOOKUP/MT UNIQUE TB TERMS
    unique_src_terms = list(set(tb['src'].tolist()))
    #get one word terms
    sst = [t for t in unique_src_terms if len(t.split())==1]
    def batch_split(l):
        max_batch_len = 10
        batches_n = int(len(l)/max_batch_len)
        steps = [n for n in range(0, len(l)+1, max_batch_len)]
        ranges = [(n, n_) for (n, n_) in list(zip(steps, steps[1:]))]
        l_batches = [l[n: n_] for (n, n_) in ranges]
        #check if there is a remainder, append it
        rem = len(l)%max_batch_len
        if rem > 0:
            l_batches = l_batches + [l[-rem:]]
            return l_batches
        else:
            return l_batches
    #make batches
    sst_batches = batch_split(sst)
    #dict look up
    sst_batches_lu = [get_azure_dict_lookup(src_det, trgdet, l) for l in sst_batches]
    #flatten list of batches
    sst_lu = [tup for list_ in sst_batches_lu for tup in list_]
    # single term dict
    lookup_dict = dict(sst_lu)
    # Multiword Source Terms
    mst = [t for t in unique_src_terms if len(t.split())>1]
    #make batches
    mst_batches = batch_split(mst)
    # azure mt
    mst_mt_batches = [get_azure_translation(l, trgdet) for l in mst_batches]
    #flatten list of batches
    mst_mt = [tup for list_ in mst_mt_batches for tup in list_]
    #multiterm dict
    mt_dict = dict(mst_mt)
    # join look up and mt dicts
    lu_and_mt_dict = {**lookup_dict, **mt_dict}
    # apply lookup translations to src column
    tb['mt_cands'] = tb['src'].apply(lambda w: lu_and_mt_dict[w])  
    tb = tb.dropna()
    return tb
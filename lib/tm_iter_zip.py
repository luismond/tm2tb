#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:03:07 2021
@author: user
Get all TM word pairs
"""
import itertools

# ITERZIP
def tm_iter_zip(tm):
    zipped_segments = list(zip(tm['src'],tm['trg']))
    tm['iter_zip'] = [iter_zip(src_segment, trg_segment) 
                      for (src_segment, trg_segment) 
                      in zipped_segments]
    return(tm)

def iter_zip(x,y):
    return list(itertools.product(x,y))
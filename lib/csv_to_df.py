#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:52:32 2021
@author: user
"""
import pandas as pd

def csv_to_df(filepath):
    tm = pd.read_csv('{}'.format(filepath), encoding='utf8')
    tm = pd.DataFrame(list(zip(tm['src'],tm['trg'])))
    tm.columns = ['src', 'trg']
    return tm
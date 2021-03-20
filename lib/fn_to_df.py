#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 05:36:27 2021
@author: user
"""
import pandas as pd
from lib.mqxliff_to_df import mqxliff_to_df
from lib.mxliff_to_df import mxliff_to_df
from lib.csv_to_df import csv_to_df
from lib.tmx_to_df import tmx_to_df

def fn_to_df(upload_path, filename):
    fileformat = filename.split('.')[-1:][0]
    filepath = '{}/{}'.format(upload_path, filename)
    # READ
    if fileformat == 'csv':
        tm = csv_to_df(filepath)
    if fileformat == 'mqxliff':
        tm = mqxliff_to_df(filepath)
    if fileformat == 'mxliff':
        tm = mxliff_to_df(filepath)
    if fileformat == 'tmx':
        tm = tmx_to_df(filepath)
    return tm

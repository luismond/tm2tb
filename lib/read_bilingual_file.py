#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilingual file reader. 
Handles different bilingual file formats: .csv, .mqxliff, .xliff, .tmx
"""
from collections import OrderedDict
import pandas as pd
import xmltodict


def read_bilingual_file(upload_path, filename):
    """
    Parameters
    ----------
    upload_path : string
        application path where uploads are temporarily stored.
    filename : string
        File name.
    Returns
    -------
    bitext : Pandas dataframe
        Bilingual file (.csv, .mxliff, .mqxliff or .tmx) converted to a two-column pandas dataframe.
    """
    fileformat = filename.split('.')[-1:][0]
    filepath = '{}/{}'.format(upload_path, filename)
    if fileformat == 'csv':
        def read_bilingual_csv(filepath):
            """
            Parameters
            ----------
            filepath : str
                The full file path obtained from the uploaded file.
            Returns
            -------
            bitext : Pandas dataframe
                Bilingual .csv file converted to a two-column pandas dataframe.
            """
            bitext = pd.read_csv('{}'.format(filepath), encoding='utf8')
            return bitext
        bitext = read_bilingual_csv(filepath)
    if fileformat == 'mqxliff':
        def read_mqxliff(filepath):
            """
            Parameters
            ----------
            filepath : str
                The full file path obtained from the uploaded file.
            Returns
            -------
            bitext : Pandas dataframe
                Bilingual .mqxliff file converted to a two-column pandas dataframe.
            """
            with open('{}'.format(filepath), encoding='utf8') as file:
                doc = xmltodict.parse(file.read())
            units = doc['xliff']['file']['body']['trans-unit']
            source_segments = [unit['source'] for unit in units]
            target_segments = [unit['target'] for unit in units]
            segments = zip(source_segments, target_segments)
            segments = [(seg[0]['#text'], seg[1]['#text']) for seg in segments
                        if '#text' in seg[0] and '#text' in seg[1]]
            bitext = pd.DataFrame(segments)
            return bitext
        bitext = read_mqxliff(filepath)
    if fileformat == 'mxliff':
        def read_mxliff(filepath):
            """
            Parameters
            ----------
            filepath : str
                The full file path obtained from the uploaded file.
            Returns
            -------
            bitext : Pandas dataframe
                Bilingual .mxliff file converted to a two-column pandas dataframe.
            """
            with open('{}'.format(filepath), encoding='utf8') as file:
                doc = xmltodict.parse(file.read())
            units = [unit['trans-unit'] for unit in doc['xliff']['file']['body']['group']]
            source_segments = [unit['source'] for unit in units]
            target_segments = [unit['target'] for unit in units]
            segments = zip(source_segments, target_segments)
            bitext = pd.DataFrame(segments)
            return bitext
        bitext = read_mxliff(filepath)
    if fileformat == 'tmx':
        def read_tmx(filepath):
            """
            Parameters
            ----------
            filepath : str
                The full file path obtained from the uploaded file.
            Returns
            -------
            bitext : Pandas dataframe
                Bilingual .tmx file converted to a two-column pandas dataframe.
            """
            with open('{}'.format(filepath)) as file:
                doc = xmltodict.parse(file.read())
            source_segments = [t['tuv'][0]['seg'] for t in doc['tmx']['body']['tu']]
            target_segments = [t['tuv'][1]['seg'] for t in doc['tmx']['body']['tu']]
            source_segments = [get_tmx_text(unit) for unit in source_segments]
            target_segments = [get_tmx_text(unit) for unit in target_segments]
            segments = zip(source_segments, target_segments)
            bitext = pd.DataFrame(segments)
            return bitext
        def get_tmx_text(unit):
            """
            Parameters
            ----------
            unit : string or OrderedDict
                Translation unit in tmx file. Can be a string or OrderedDict.
            Returns
            -------
            unit_text : string
                Raw text from translation unit or empty string in case of exception.
            """
            if isinstance(unit, str) is True:
                unit_text = unit
                return unit_text
            if isinstance(unit, OrderedDict) is True:
                unit_text = unit['#text']
                return unit_text
            return None
        bitext = read_tmx(filepath)
    #Final steps
    bitext.columns = ['src', 'trg']
    bitext = bitext.fillna('')
    bitext = bitext[bitext['src'].str.len()>0]
    bitext = bitext[bitext['trg'].str.len()>0]
    return bitext



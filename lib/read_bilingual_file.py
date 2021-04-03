#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilingual file reader.
Handles different bilingual file formats: .csv, .mqxliff, .xliff, .tmx
"""
from collections import OrderedDict
import pandas as pd
import xmltodict

class BilingualReader():
    """
    A class to read bilingual files
    Atributes
    ---------
    upload_path: string
        upload path of the application
    filename: string
        name of the file
    fileformat: string
        format of the file. Can be .csv, .mqxliff, .mxliff or .tmx
    filepath: string
        full file path of the file
    columns: list
        column names for the bilingual text
    Methods
    -------
    read_csv():
        Reads the file if the fileformat is .csv
    read_mqxliff():
        Reads the file if the fileformat is .mqxliff
    read_mxliff():
        Reads the file if the fileformat is .mxliff
    read_tmx():
        Reads the file if the fileformat is .tmx
    get_bitext():
        Gets the file format and calls the corresponding format reader to get the bilingual text
    finalize(bitext):
        Adds the column names to the bilingual text, drops empty cells
    """
    def __init__(self, upload_path, filename):
        """
        Parameters
        ----------
        upload_path : string
            upload path of the app.
        filename : string
            name of the file.
        Returns
        -------
        None.
        """
        self.upload_path = upload_path
        self.filename = filename
        self.fileformat = filename.split('.')[-1:][0]
        self.filepath = '{}/{}'.format(upload_path, filename)
        self.columns = ['source', 'target']

    def finalize(self, bitext):
        """
        Parameters
        ----------
        bitext : Pandas dataframe
            Two column bilingual dataframe
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe with empty cells removed and named columns
        """
        bitext.columns = self.columns
        bitext = bitext.astype(str)
        bitext = bitext.fillna('')
        bitext = bitext[bitext['source'].str.len()>0]
        bitext = bitext[bitext['target'].str.len()>0]
        return bitext

    def read_csv(self):
        """
        Raises
        ------
        ValueError
            If the csv containes less or more than 2 columns.
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe
        """
        bitext = pd.read_csv('{}'.format(self.filepath), encoding='utf8')
        if len(bitext.columns) == 2:
            bitext = self.finalize(bitext)
        else:
            raise ValueError("Your .csv file must have 1 'source' column and 1 'target' column.")
        return bitext

    def read_mqxliff(self):
        """
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe
        """
        with open('{}'.format(self.filepath), encoding='utf8') as file:
            doc = xmltodict.parse(file.read())
        units = doc['xliff']['file']['body']['trans-unit']
        source_segments = [unit['source'] for unit in units]
        target_segments = [unit['target'] for unit in units]
        segments = zip(source_segments, target_segments)
        segments = [(seg[0]['#text'], seg[1]['#text']) for seg in segments
                    if '#text' in seg[0] and '#text' in seg[1]]
        bitext = pd.DataFrame(segments)
        bitext = self.finalize(bitext)
        return bitext

    def read_mxliff(self):
        """
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe
        """
        with open('{}'.format(self.filepath), encoding='utf8') as file:
            doc = xmltodict.parse(file.read())
        units = [unit['trans-unit'] for unit in doc['xliff']['file']['body']['group']]
        source_segments = [unit['source'] for unit in units]
        target_segments = [unit['target'] for unit in units]
        segments = zip(source_segments, target_segments)
        bitext = pd.DataFrame(segments)
        bitext = self.finalize(bitext)
        return bitext

    def read_tmx(self):
        """
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe
        """
        with open('{}'.format(self.filepath)) as file:
            doc = xmltodict.parse(file.read())
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
        source_segments = [t['tuv'][0]['seg'] for t in doc['tmx']['body']['tu']]
        target_segments = [t['tuv'][1]['seg'] for t in doc['tmx']['body']['tu']]
        source_segments = [get_tmx_text(unit) for unit in source_segments]
        target_segments = [get_tmx_text(unit) for unit in target_segments]
        segments = zip(source_segments, target_segments)
        bitext = pd.DataFrame(segments)
        bitext = self.finalize(bitext)
        return bitext

    def get_bitext(self):
        """
        Raises
        ------
        NameError
            The format file is not supported.
        Returns
        -------
        bitext : Pandas dataframe
            Two column bilingual dataframe, ready for further processing
        """
        print(self.fileformat)
        if self.fileformat == 'csv':
            bitext = self.read_csv()
        if self.fileformat == 'mqxliff':
            bitext = self.read_mqxliff()
        if self.fileformat == 'mxliff':
            bitext = self.read_mxliff()
        if self.fileformat == 'tmx':
            bitext = self.read_tmx()
        return bitext


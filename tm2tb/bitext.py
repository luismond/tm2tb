#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiText class
"""
import os
import json
from collections import Counter as cnt
import pandas as pd
import requests

from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import BilingualReader
#from tm2tb import DistanceApi

class BiText:
    """
    A class to represent a bilingual document.
    (In translator's terms known as a "Translation Memory" or "bilingual document").

    Extracts bilingual ngrams from a bilingual text.
    (In translator's terms known as a "Term Base").

    This is the class that gets a TB from a TM :)


    Attributes
    ----------
    path : str
        Path of the bilingual file.

    file_name : str
        File name of the bilingual file.

    bitext_doc : Pandas dataframe
        Content of the bilingual file

    Methods
    -------
    get_bitext_doc()
        Gets the contents of the bilingual file.

    get_bitext_bilingual_ngrams_precise()
        Gets a collection of bilingual ngrams from the bilingual document.

    get_bitext_bilingual_ngrams_fast()
        A faster method for getting a collection of bilingual ngrams
        from the bilingual document.

    """

    def __init__(self, path, file_name):
        self.path = path
        self.file_name = file_name
        self.bitext = self.get_bitext_doc()

    def get_bitext_doc(self):
        """
        Returns
        -------
        bitext : Pandas dataframe
            Two column dataframe.
                Column 1: "src". Represents a collection of sentences.
                Column 2: "trg". Represents a collection of translated sentences.
        """
        bitext = BilingualReader(self.path, self.file_name).get_bitext()
        return bitext

    def get_bitext_bilingual_ngrams_precise(self,
                                         diversity=.5,
                                         top_n=8,
                                         min_similarity=.8,
                                         **kwargs):
        """
        Iterate over all the bilingual sentences, extract bilingual ngrams.
        (See BiSentence class).
        This method is slower, but more precise.

        Parameters
        ----------
        diversity : TYPE, optional
            DESCRIPTION. The default is .5.
        top_n : TYPE, optional
            DESCRIPTION. The default is 8.
        min_seq_to_seq_sim : float, optional
            DESCRIPTION. The default is .8.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        bitext_bilingual_ngrams : Pandas dataframe
            DESCRIPTION. columns: src_ngram, trg_ngram, similarity

        """

        all_bilingual_ngrams = []
        for i in range(len(self.bitext)):
            try:
                src_row = self.bitext.iloc[i]['src']
                trg_row = self.bitext.iloc[i]['trg']
                bisentence = BiSentence(src_row, trg_row)
                bilingual_ngrams = bisentence.get_bilingual_ngrams(
                                                        diversity=diversity,
                                                        top_n=top_n,
                                                        **kwargs)
                all_bilingual_ngrams.append(bilingual_ngrams)
            except:
                pass
        if len(all_bilingual_ngrams)==0:
            raise ValueError("No bitext_bilingual_ngrams from get_bitext_bilingual_ngrams_precise")

        bitext_bilingual_ngrams = pd.concat(all_bilingual_ngrams)
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.drop_duplicates(subset='src')
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.reset_index()
        bitext_bilingual_ngrams = self.filter_bilingual_ngrams(bitext_bilingual_ngrams,
                                                               min_similarity)
        return bitext_bilingual_ngrams

    def get_bitext_bilingual_ngrams_fast(self,
                                         diversity=.5,
                                         top_n=8,
                                         min_similarity=.8,
                                         server_mode="local",
                                         **kwargs):

        """
        Instead of extracting bilingual ngrams from each sentence pair,
        get all source ngrams, all trg ngrams, and query their similarity.
        This method is faster, but less precise.

        Parameters
        ----------
        diversity : TYPE, optional
            DESCRIPTION. The default is .5.
        top_n : TYPE, optional
            DESCRIPTION. The default is 8.
        min_seq_to_seq_sim : TYPE, optional
            DESCRIPTION. The default is .8.
        server_mode : TYPE, optional
            DESCRIPTION. The default is "local".
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        bitext_bilingual_ngrams : Pandas dataframe
            DESCRIPTION. columns: src_ngram, trg_ngram, similarity

        """

        # get all src & trg ngrams
        def get_src_trg_top_ngrams():
            bitext_src_top_ngrams = []
            bitext_trg_top_ngrams = []
            for i in range(len(self.bitext)):
                try:
                    src_sentence = Sentence(self.bitext.iloc[i]['src'])
                    src_top_ngrams, _ = zip(*src_sentence.get_top_ngrams(**kwargs))
                    for ngram in src_top_ngrams:
                        if not ngram in bitext_src_top_ngrams:
                            bitext_src_top_ngrams.append(ngram)

                    trg_sentence = Sentence(self.bitext.iloc[i]['trg'])
                    trg_top_ngrams, _ = zip(*trg_sentence.get_top_ngrams(**kwargs))
                    for ngram in trg_top_ngrams:
                        if not ngram in bitext_trg_top_ngrams:
                            bitext_trg_top_ngrams.append(ngram)
                except:
                    pass
            if len(bitext_src_top_ngrams) == 0 or len(bitext_trg_top_ngrams)==0:
                raise ValueError('No bitext_bilingual_ngrams from get_bitext_bilingual_ngrams_fast')
            return bitext_src_top_ngrams, bitext_trg_top_ngrams

        bitext_src_top_ngrams, bitext_trg_top_ngrams = get_src_trg_top_ngrams()

        # Get src ngrams & trg ngrams similarity
        seq1 = bitext_src_top_ngrams
        seq2 = bitext_trg_top_ngrams
        params = json.dumps(
                {
                    'seq1':seq1,
                    'seq2':seq2,
                    'diversity':diversity,#todo: remove
                    'top_n':top_n,
                    'query_type':'src_ngrams_to_trg_ngrams'
                })
        if server_mode=='remote':
            url = 'http://0.0.0.0:5000/distance_api'
            response = requests.post(url=url, json=params).json()
        if server_mode=='local':
            response = DistanceApi(params).post()
        max_seq_similarities = json.loads(response)

        # filter by similarity and return best bilingual_ngrams
        bitext_bilingual_ngrams = pd.DataFrame(max_seq_similarities)
        bitext_bilingual_ngrams.columns = ['src', 'trg', 'similarity']
        bitext_bilingual_ngrams = self.filter_bilingual_ngrams(bitext_bilingual_ngrams,
                                                               min_similarity)
        return bitext_bilingual_ngrams

    def filter_bilingual_ngrams(self, bilingual_ngrams, min_similarity):
        'Filters bilingual ngrams'
        # Group by source ngram, get most similar target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target ngram, get most similar source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by similarity
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['similarity'] >= min_similarity]
        return bilingual_ngrams

    def save_biterms(self, biterms):
        'Save extracted biterms'
        biterms_path = os.path.join(self.path, '{}_tb.csv'.format(self.file_name))
        biterms.to_csv(biterms_path, encoding='utf-8-sig', index=False)

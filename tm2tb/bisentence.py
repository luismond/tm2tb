#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiSentence class
"""
import json
import requests
import pandas as pd

from tm2tb import Sentence
#from tm2tb import DistanceApi

class BiSentence:
    """
    A class to represent a source sentence, a target sentence, and their ngrams.
    Extracts from both sentences.
    Aligns extracted ngrams (returning "bilingual ngrams").

    Attributes
    ----------
    src_sentence : str
        Raw Unicode sentence, short text or paragraph.

    trg_sentence : str
        Raw Unicode sentence, short text or paragraph.

    Methods
    -------
    get_src_ngrams(**kwargs)
        Gets ngrams from the source sentence.

    get_trg_ngrams(**kwargs)
        Gets ngrams from the target sentence.

    get_bilingual_ngrams(min_distance=, server_mode=, **kwargs)
        Gets aligned bilingual ngrams from the source sentence & the target sentence.

    """

    def __init__(self, src_sentence, trg_sentence):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence

    def get_src_ngrams(self, **kwargs):
        'Gets best ngrams from target sentence'
        sentence = Sentence(self.src_sentence)
        src_ngrams = sentence.get_top_ngrams(**kwargs)
        return src_ngrams

    def get_trg_ngrams(self, **kwargs):
        'Gets best ngrams from target sentence'
        sentence = Sentence(self.trg_sentence)
        trg_ngrams = sentence.get_top_ngrams(**kwargs)
        return trg_ngrams

    def get_bilingual_ngrams(self,
                             min_similarity=.7,
                             server_mode='local',
                             **kwargs):
        """

        Aligns source ngrams and target ngrams

        Parameters
        ----------
        min_similarity : float, optional
            DESCRIPTION. Minimum similarity value between source and target ngrams.
                         The default is .7.
        server_mode : string, optional
            DESCRIPTION. Defines if the similarity queries are done locally or remotely.
        **kwargs : dict
            DESCRIPTION. Optional parameters passed to the Sentence objects.


        Raises
        ------
        ValueError
            If no similar ngrams are found, raises ValueError.

        Returns
        -------
        bilingual_ngrams : Pandas dataframe
            DESCRIPTION. columns: src_ngram, trg_ngram, similarity

        """

        #Get src & trg ngrams
        src_ngrams, _ = zip(*self.get_src_ngrams(**kwargs))
        trg_ngrams, _ = zip(*self.get_trg_ngrams(**kwargs))

        # Sends src & trg ngrams to distance api
        params = json.dumps(
                {'seq1':src_ngrams,
                'seq2':trg_ngrams,
                'diversity':.5,
                'top_n':28,
                'query_type':'src_ngrams_to_trg_ngrams'})

        if server_mode=='remote':
            url = 'http://0.0.0.0:5000/distance_api'
            response = requests.post(url=url, json=params).json()

        if server_mode=='local':
            response = DistanceApi(params).post()

        bilingual_ngrams_distances = json.loads(response)

        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(bilingual_ngrams_distances)
        bilingual_ngrams.columns = ['src', 'trg', 'similarity']

        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by distance
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['similarity'] >= min_similarity]
        bilingual_ngrams = bilingual_ngrams.sort_values(by='similarity')

        # Validate bisentence
        if len(bilingual_ngrams)==0:
            raise ValueError('No similar bilingual_ngrams found!')

        return bilingual_ngrams

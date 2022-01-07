#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiSentence class
"""
import json
import requests
import pandas as pd
from tm2tb import Sentence
#from tm2tb import SimilarityApi

class BiSentence:
    """
    Takes a source sentence and a target sentence.
    Gets ngrams from both sentences.
    Compares ngrams to find the ngram pairs that are translations of each other.
    """
    def __init__(self, src_sentence, trg_sentence):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence

    def get_src_ngrams(self, **kwargs):
        'Gets best ngrams from source sentence'
        sn = Sentence(self.src_sentence)
        src_ngrams = sn.get_non_overlapping_ngrams(**kwargs)
        return [a for (a,b) in src_ngrams]

    def get_trg_ngrams(self, **kwargs):
        'Gets best ngrams from target sentence'
        sn = Sentence(self.trg_sentence)
        trg_ngrams = sn.get_non_overlapping_ngrams(**kwargs)
        return [a for (a,b) in trg_ngrams]

    def get_bilingual_ngrams(self,
                             min_distance=.44,
                             distance_api_mode='remote',
                             **kwargs):
        'Aligns source and target ngrams'

        #Get src & trg ngrams
        src_ngrams = self.get_src_ngrams(**kwargs)
        trg_ngrams = self.get_trg_ngrams(**kwargs)

        # Sends src & trg ngrams to distance api
        params = json.dumps(
                {'seq1':trg_ngrams,
                'seq2':src_ngrams,
                'diversity':.5,
                'top_n':8,#todo: remove
                'query_type':'src_ngrams_to_trg_ngrams'})

        url = 'http://0.0.0.0:5000/distance_api'
        response = requests.post(url=url, json=params).json()

        if distance_api_mode=='remote':
            bilingual_ngrams_distances = json.loads(response)

        if distance_api_mode=='local':
            bilingual_ngrams_distances = SimilarityApi(params).get_closest_sequence_elements()

        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(bilingual_ngrams_distances)
        bilingual_ngrams.columns = ['src', 'trg', 'distance']

        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by distance
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['distance'] <= min_distance]

        # Validate bisentence
        if len(bilingual_ngrams)==0:
            raise ValueError('No similar bilingual_ngrams found!')

        return bilingual_ngrams

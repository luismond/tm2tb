#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiSentence class
"""
import json
import requests
import pandas as pd
from tm2tb import Sentence

class BiSentence:
    """
    Takes a source sentence and a target sentence.
    Gets ngrams from both sentences.
    Compares ngrams to find the ngram pairs that are translations of each other.
    """
    def __init__(self, src_sentence, trg_sentence, **kwargs):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence
        self.min_distance = .44
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1

        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3

        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']

    def get_sentence_ngrams(self, sentence):
        sn = Sentence(sentence,
                        ngrams_min = self.ngrams_min,
                        ngrams_max = self.ngrams_max,
                        good_tags = self.good_tags)

        non_overlapping_ngrams = sn.get_non_overlapping_ngrams()
        return [a for (a, b) in non_overlapping_ngrams]


    def get_src_ngrams(self):
        src_ngrams = self.get_sentence_ngrams(self.src_sentence)
        return src_ngrams
    
    def get_trg_ngrams(self):
        trg_ngrams = self.get_sentence_ngrams(self.trg_sentence)
        return trg_ngrams

    def get_bilingual_ngrams_distances(self):
        """
        Fetches src and trg ngrams.
        Sends them to /sim_api to get their distances.
        """
        src_ngrams = self.get_src_ngrams()
        trg_ngrams = self.get_trg_ngrams()
        url = 'http://0.0.0.0:5000/sim_api'
        params = json.dumps({
            'seq1':src_ngrams,
            'seq2':trg_ngrams})
        
        response = requests.post(url=url, json=params).json()
        bilingual_ngrams_distances = json.loads(response)

        return bilingual_ngrams_distances

    
    def filter_bilingual_ngrams(self):
        bnd = self.get_bilingual_ngrams_distances()
        
        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(bnd)
        bilingual_ngrams.columns = ['src', 'trg', 'distance']

        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by distance
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['distance'] <= self.min_distance]

        # # Validate bilingual_ngrams dataframe
        if len(bilingual_ngrams)==0:
            raise ValueError('No similar bilingual_ngrams found!')
            
        return bilingual_ngrams
    

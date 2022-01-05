#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiSentence class
"""
import json
import requests
from langdetect import detect
import pandas as pd
from tm2tb import Sentence
from tm2tb import Ngrams

class BiSentence:
    """
    Initialized with a source sentence and a target sentence.
    Implements methods for comparing a source sentence and a target sentence.
    It is assumed that the two sentences are translations of each other.
    """
    def __init__(self, src_sentence, trg_sentence, **kwargs):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence
        self.min_distance = .5
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1

        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3

        if 'src_lang' in kwargs.keys():
            self.src_lang = kwargs.get('src_lang')
        else:
            self.src_lang = detect(self.src_sentence)

        if 'trg_lang' in kwargs.keys():
            self.trg_lang = kwargs.get('trg_lang')
        else:
            self.trg_lang = detect(self.trg_sentence)

        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']

    def get_candidate_ngrams(self, sentence):
        """
        Parameters
        ----------
        sentence : string
            String representing a sentence or paragraph.

        Returns
        -------
        candidate_ngrams : list
            List of substrings representing terms/ngrams/keywords from sentence.
        """
        sn = Sentence(sentence,
                        ngrams_min = self.ngrams_min,
                        ngrams_max = self.ngrams_max,
                        good_tags = self.good_tags)
        
        sentence = sn.get_clean_sentence()
        lang = sn.lang
        ng = Ngrams(sentence, lang=lang)
        
        candidate_ngrams = ng.get_candidate_ngrams()
        
        #candidate_ngrams = candidate_ngrams[candidate_ngrams['v']==1]
        candidate_ngrams = candidate_ngrams['joined_ngrams'].tolist()
        return candidate_ngrams

    def get_bilingual_ngrams(self):
        """
        Get src ngrams.
        Get trg ngrams.
        Get their distances.
        Use conditions to filter them.
        """
        src_candidate_ngrams = self.get_candidate_ngrams(self.src_sentence)
        trg_candidate_ngrams = self.get_candidate_ngrams(self.trg_sentence)
       
        def get_src_trg_ngram_distances(src_candidate_ngrams, trg_candidate_ngrams):
            'Get distances from src ngrams and trg ngrams'
            url = 'http://0.0.0.0:5000/sim_api'
            params = json.dumps({
                'seq1':src_candidate_ngrams,
                'seq2':trg_candidate_ngrams})
            response = requests.post(url=url, json=params).json()
            data = json.loads(response)
            if len(data)==0:
                raise ValueError('No similar terms found')
            return data
        
        bilingual_ngrams = get_src_trg_ngram_distances(src_candidate_ngrams,
                                                       trg_candidate_ngrams)
        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(bilingual_ngrams)
        bilingual_ngrams.columns = ['src', 'trg', 'distance']
        
        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])
        
        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['distance'].idxmin()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])
        
        # Filter by distance
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['distance'] <= self.min_distance]
        
        # Validate bisentence
        if len(bilingual_ngrams)==0:
            raise ValueError('No similar terms found')
        return bilingual_ngrams

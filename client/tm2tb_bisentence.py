#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB BiSentence class
"""
import json
import requests
from langdetect import detect
from tm2tb_sentence import Sentence

class BiSentence:
    'Represents a pair of Sentence objects'
    def __init__(self, src_sentence, trg_sentence, **kwargs):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence

        'Get optional arguments or use default arguments'
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

    def get_src_cands(self):
        'Return candidate ngrams from source sentence'
        sentence_object = Sentence(self.src_sentence,
                            lang=self.src_lang,
                            ngrams_min = self.ngrams_min,
                            ngrams_max = self.ngrams_max,
                            good_tags = self.good_tags)
        src_cands = sentence_object.get_term_candidates()
        return src_cands

    def get_trg_cands(self):
        'Return candidate ngrams from target sentence'
        sentence_object = Sentence(self.trg_sentence,
                            lang=self.trg_lang,
                            ngrams_min = self.ngrams_min,
                            ngrams_max = self.ngrams_max,
                            good_tags = self.good_tags)
        trg_cands = sentence_object.get_term_candidates()
        return trg_cands

    def get_terms_similarity(self):
        'Compare src and trg cands, get similarities'
        src_cands = self.get_src_cands()
        trg_cands = self.get_trg_cands()
        url = 'url'
        params = json.dumps({
            'src_cands':src_cands,
            'trg_cands':trg_cands})
        response = requests.post(url=url, json=params).json()
        data = json.loads(response)
        return data

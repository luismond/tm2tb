#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence class
"""
from langdetect import detect
import pandas as pd

import es_core_news_sm
import en_core_web_sm
import de_core_news_sm
import pt_core_news_sm
model_en = en_core_web_sm.load()
model_es = es_core_news_sm.load()
model_de = de_core_news_sm.load()
model_pt = pt_core_news_sm.load()
spacy_models = {'model_en':model_en,
                'model_es':model_es,
                'model_de':model_de,
                'model_pt':model_pt}

class Sentence:
    'Sentence class. Instantiates a sentence object from a string sentence'
    supported_languages = ['en', 'es', 'de', 'pt']
    def __init__(self, sentence, **kwargs):
        self.sentence = self.normalize_space(sentence)
        'Get optional arguments or use default arguments'
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1
        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3
        if 'lang' in kwargs.keys():
            self.lang = kwargs.get('lang')
        else:
            self.lang = detect(self.sentence)
        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']

    def normalize_space(self, sentence):
        'Normalizes whitespace'
        ords = [9, 10, 13, 32, 160]
        for char in sentence:
            if ord(char) in ords:
                sentence = sentence.replace(char, ' ')
        return sentence

    def get_doc(self):
        'Instantiate a spacy sentence object from raw text'
        spacy_model = spacy_models['model_{}'.format(self.lang)]
        return spacy_model(self.sentence)

    def get_tokens(self):
        'Get token and part-of-speech from each token in sentence'
        return [token.text for token in self.get_doc()]

    def get_pos_tagged_tokens(self):
        'Get token and part-of-speech from each token in sentence'
        return [(token.text, token.pos_) for token in self.get_doc()]

    def get_ngrams(self, seq):
        'Get n-grams from any sequence'
        min_ = self.ngrams_min
        max_ = self.ngrams_max
        ngrams = [list(zip(*[seq[i:] for i in range(n)]))
                  for n in range(min_, max_+1)]
        return [ng for ngl in ngrams for ng in ngl]

    def get_token_ngrams(self):
        'Get n-grams from sentence tokens'
        return self.get_ngrams(self.get_tokens())

    def get_pos_tagged_ngrams(self):
        'Get n-grams from pos-tagged tokens'
        return self.get_ngrams(self.get_pos_tagged_tokens())

    def filter_pos_tagged_ngrams(self):
        'Filter pos-tagged ngrams to get candidate terms'
        ptn = self.get_pos_tagged_ngrams()
        good_tags = self.good_tags
        #keep ngrams with good tags at start and end
        ptn = list(filter(lambda tl: tl[0][1] in good_tags
                          and tl[-1:][0][1] in good_tags, ptn))
        #drop ngrams with punctuation
        ptn = list(filter(lambda tl: any(t=='PUNCT' for t in
                                         [t for (tk, t) in tl])==False, ptn))
        return ptn

    def get_term_candidates(self):
        'Return candidates from the filtered pos-tagged ngrams as strings'
        fptn = self.filter_pos_tagged_ngrams()
        cands = [' '.join([token for (token, tag) in tuple_list])
                for tuple_list in fptn]
        cands = [t.replace(" 's", "'s") for t in cands]
        return cands

    def get_ngrams_df(self, seq):
        '''Create a pandas table from the tokens and their pos-tags.
        For visualization purposes'''
        tokens = [[token for (token, tag) in tup_list] for tup_list in seq]
        tags = [[tag for (token, tag) in tup_list] for tup_list in seq]
        ngrams_df = pd.DataFrame(list(zip(tokens, tags)))
        ngrams_df.columns = ['tokens','tags']
        return ngrams_df
    

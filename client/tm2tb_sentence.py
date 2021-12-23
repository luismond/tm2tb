#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence class
"""
import re
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
        self.sentence = self.clean_sentence(sentence)
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

    def clean_sentence(self, sentence):
        'Cleans sentence'
        def normalize_space_chars(sentence):
            'Replaces weird spaces with normal space'
            ords = [9, 10, 13, 32, 160]
            for char in sentence:
                if ord(char) in ords:
                    sentence = sentence.replace(char, ' ')
            return sentence

        def normalize_space_seqs(sentence):
            'Finds sequences of more than one space, returns one space'
            def repl(match):
                return ' '
            sentence = re.sub(r"(\s+)", repl, sentence)
            return sentence

        def normalize_apostrophe(sentence):
            'Replace curved apostrophe with straight apostrophe'
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"'s", groups[2])
            pattern = r"(.|\s)(’s)(.|\s)"
            return re.sub(pattern, repl, sentence)

        def normalize_newline(sentence):
            'Replaces hard coded newlines with normal newline symbol'
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"\n", groups[2])
            pattern = r"(.)(\n|\\n|\\\n|\\\\n|\\\\\n)(.)"
            return re.sub(pattern, repl, sentence)

        sentence = normalize_space_chars(sentence)
        sentence = normalize_space_seqs(sentence)
        sentence = normalize_apostrophe(sentence)
        sentence = normalize_newline(sentence)
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
        ptn = list(filter(lambda tl: tl[0][0].isalpha()
                          and tl[-1:][0][0].isalpha(), ptn))
        # certain punctuation symbols not allowed in the term
        bad_punct = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
               '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        ptn = list(filter(lambda tl: any(t[0] in bad_punct for t in tl) is False, ptn))
        return ptn

    def get_term_candidates(self):
        'Return candidates from the filtered pos-tagged ngrams as strings'
        fptn = self.filter_pos_tagged_ngrams()
        cands = [' '.join([token for (token, tag) in tuple_list])
                for tuple_list in fptn]

        def rejoin_split_punct(string):
            'rejoin second position punct char to first position token'
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®)(.+)"
            return re.sub(pattern, repl, string)

        cands = [rejoin_split_punct(t) for t in cands]

        if len(cands)==0:
            raise ValueError('No term candidates found')
        if len(cands)==1:
            raise ValueError('Only one candidate found')
        return cands

    def get_ngrams_df(self, seq):
        '''Create a pandas table from the tokens and their pos-tags.
        For visualization purposes'''
        tokens = [[token for (token, tag) in tup_list] for tup_list in seq]
        tags = [[tag for (token, tag) in tup_list] for tup_list in seq]
        ngrams_df = pd.DataFrame(list(zip(tokens, tags)))
        ngrams_df.columns = ['tokens','tags']
        return ngrams_df
    

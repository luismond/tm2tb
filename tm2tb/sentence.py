#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB Sentence class.
Implements methods for string cleaning, validation, tokenization,
ngram generation and ngram selection.
"""
import re
import json
from langdetect import detect
import requests

import es_dep_news_trf
import en_core_web_trf
#import en_core_web_sm
import de_dep_news_trf
import fr_dep_news_trf

#from tm2tb import DistanceApi
#model_en = en_core_web_sm.load()
model_en = en_core_web_trf.load()
model_es = es_dep_news_trf.load()
model_de = de_dep_news_trf.load()
model_fr = fr_dep_news_trf.load()


class Sentence:
    """
    Takes a string representing a sentence.
    Instantiates a sentence object.
    Implements methods to preprocess and validate a sentence.
    Returns a clean sentence.
    """
    supported_languages = ['en', 'es', 'de', 'fr']

    def __init__(self, sentence, **kwargs):
        self.sentence = sentence
        self.kwargs = kwargs
        self.clean_sentence = self.preprocess()

    def preprocess(self,
                   min_non_alpha_ratio = .25,
                   sentence_min_length = 100,
                   sentence_max_length = 900):
        """
        Normalizes spaces, apostrophes and special characters.
        Validates sentence alphabetic-ratio, length, and language.
        """
        def normalize_space_chars(sentence):
            """
            Replaces all spaces with normal spaces.
            """
            ords = [9, 10, 13, 32, 160]
            for char in sentence:
                if ord(char) in ords:
                    sentence = sentence.replace(char, ' ')
            return sentence

        def normalize_space_seqs(sentence):
            """
            Finds sequences of more than one space, returns one space.
            """
            def repl(match):
                return ' '
            sentence = re.sub(r"(\s+)", repl, sentence)
            return sentence

        def normalize_apostrophe(sentence):
            """
            Replaces curved apostrophe with straight apostrophe.
            """
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"'s", groups[2])
            pattern = r"(.|\s)(’s)(.|\s)"
            return re.sub(pattern, repl, sentence)

        def normalize_newline(sentence):
            """
            Replaces hard coded newlines with normal newline symbol.
            """
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"\n", groups[2])
            pattern = r"(.)(\n|\\n|\\\n|\\\\n|\\\\\n)(.)"
            return re.sub(pattern, repl, sentence)

        def validate_if_mostly_alphabetic(sentence):
            """
            Checks if most of the characters in sentence are alphabetic.
            """
            alpha = len([char for char in sentence if char.isalpha()])
            if alpha==0:
                raise ValueError('No alphanumeric chars found!')
            non_alpha = len([char for char in sentence
                             if not char.isalpha() and not char==' '])
            non_alpha_ratio = non_alpha/alpha
            if non_alpha_ratio >= min_non_alpha_ratio:
                raise ValueError('Too many non-alpha chars!')
            if sentence.startswith('http'):
                raise ValueError('Cannot process http addresses!')
            if sentence.isdigit():
                raise ValueError('Sentence contains only numbers!')
            return sentence

        def validate_length(sentence):
            """
            Checks if sentence length is between min and max length values.
            """
            if len(sentence) <= sentence_min_length:
                raise ValueError('Sentence is too short!')
            if len(sentence) >= sentence_max_length:
                raise ValueError('Sentence is too long!')
            return sentence

        def validate_lang(sentence):
            """
            Checks if sentence language is supported.
            """
            if 'lang' in self.kwargs.keys():
                self.lang = self.kwargs.get('lang')
            else:
                self.lang = detect(sentence)
            if self.lang not in self.supported_languages:
                raise ValueError('Language not supported!')
            return sentence

        sentence = normalize_space_chars(self.sentence)
        sentence = normalize_space_seqs(sentence)
        sentence = normalize_apostrophe(sentence)
        sentence = normalize_newline(sentence)
        sentence = validate_if_mostly_alphabetic(sentence)
        sentence = validate_length(sentence)
        sentence = validate_lang(sentence)
        return sentence

    def get_spacy_model(self):
        'Gets spacy model'
        if self.lang=='en':
            spacy_model = model_en
        if self.lang=='es':
            spacy_model = model_es
        if self.lang=='de':
            spacy_model = model_de
        if self.lang=='fr':
            spacy_model = model_fr
        return spacy_model

    def get_ngrams(self,
                   ngrams_min = 1,
                   ngrams_max = 3,
                   include_pos = ['NOUN','PROPN'],
                   exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX']):
        'Get ngrams with filtering options'
        #include_punct = ["'", ":", "’", "’", "'", "™", "®", "%"]
        exclude_punct = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
                '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']

        spacy_model = self.get_spacy_model()
        doc = spacy_model(self.clean_sentence)

        # Get text and part-of-speech tag for each token in document
        pos_tokens = [(token.text, token.pos_) for token in doc]

        # Get ngrams from pos_tokens
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))

        pos_ngrams = (ng for ngl in pos_ngrams for ng in ngl)

        # Keep ngrams where the first element's pos-tag
        # and the last element's pos-tag are present in include_pos
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][1] in include_pos
                          and pos_ngram[-1:][0][1] in include_pos, pos_ngrams)

        # Keep ngrams where the first element's token
        # and the last element's token are alpha
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][0].isalpha()
                          and pos_ngram[-1:][0][0].isalpha(), pos_ngrams)

        # Keep ngrams where none of elements' tag is in exclude pos
        pos_ngrams = filter(lambda pos_ngram: not any(token[1] in exclude_pos
                                                      for token in pos_ngram), pos_ngrams)

        # Keep ngrams where any of the middle elements' text is in exclude punct
        pos_ngrams = filter(lambda pos_ngram: not any((token[0] in exclude_punct
                                                       for token in pos_ngram[1:-1])), pos_ngrams)

        def rejoin_special_punct(ngram):
            'Joins apostrophes and other special characters to their token.'
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
            return re.sub(pattern, repl, ngram)

        result = {'ngrams':[],
                  'tags':[],
                  'joined_ngrams':[]}

        for pos_ngram in pos_ngrams:
            ngram, tag = zip(*pos_ngram)
            joined_ngram = rejoin_special_punct(' '.join(ngram))
            result['ngrams'].append(ngram)
            result['joined_ngrams'].append(joined_ngram)
            result['tags'].append(tag)

        return result


    def get_top_ngrams(self,
                   server_mode='remote',
                   diversity=.8,
                   top_n=50,
                   overlap=True,
                   **kwargs):
        'Get those ngrams that are most similar to the sentence'

        sentence = self.clean_sentence
        ngrams = self.get_ngrams(**kwargs)
        ngrams = list(set(ngrams['joined_ngrams']))

        params = json.dumps(
            {'seq1':[sentence],
             'seq2':ngrams,
             'diversity':diversity,
             'top_n':top_n,
             'query_type':'ngrams_to_sentence'})

        url = 'http://0.0.0.0:5000/distance_api'
        response = requests.post(url=url, json=params).json()
        top_ngrams = [tuple(el) for el in json.loads(response)]

        # if server_mode=='local':
        #     top_ngrams = DistanceApi(params).get_top_ngrams()

        # Look for top_ngrams in sentence, remove them from sentence.
        # If top_ngram not present in sentence, remove it from top_ngrams.
        if overlap is False:
            for tup in top_ngrams:
                pattern = r"(^|\s|\W)({})($|\s|\W)".format(tup[0])
                matches = re.findall(pattern, sentence)
                sentence = re.sub(pattern, ' ', sentence)
                if len(matches)==0:
                    top_ngrams.remove(tup)
        if overlap is True:
            pass
        return top_ngrams

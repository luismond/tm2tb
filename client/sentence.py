#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB Sentence class. Implements string cleaning and validation methods.
"""
import re
from langdetect import detect

class Sentence:
    """
    Takes a string representing a sentence.
    Instantiates a sentence object from a string representing a sentence.
    Implements methods to minimally clean and validate a string representing a sentence.
    Returns a minimally cleaned string representing a sentence.
    """
    supported_languages = ['en', 'es', 'de', 'pt', 'fr']
    def __init__(self, sentence, **kwargs):
        self.sentence = sentence
        self.kwargs = kwargs
        self.sentence_min_length = 40
        self.sentence_max_length = 400
        self.min_non_alpha_ratio = .25

        #todo: find a better way to pass kwargs
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
        if 'bad_tags' in kwargs.keys():
            self.bad_tags = kwargs.get('bad_tags')
        else:
            self.bad_tags = ['X', 'SCONJ', 'CCONJ', 'VERB']

    def get_clean_sentence(self):
        sentence = self.sentence
        """
        Performs cleaning and validation operations
        before attempting to return string representing a sentence.
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

        def validate_sentence(sentence):
            """
            Validates a sequence of characters representing a sentence.
            """
            alpha = len([char for char in sentence if char.isalpha()])
            if alpha==0:
                raise ValueError('No alphanumeric chars found!')
            non_alpha = len([char for char in sentence
                             if not char.isalpha() and not char==' '])
            non_alpha_ratio = non_alpha/alpha
            if non_alpha_ratio >= self.min_non_alpha_ratio:
                raise ValueError('Too many non-alpha chars!')
            if len(sentence) <= self.sentence_min_length:
                raise ValueError('Sentence is too short!')
            if len(sentence) >= self.sentence_max_length:
                raise ValueError('Sentence is too long!')
            if 'http' in sentence:
                raise ValueError('Cannot process http addresses!')
            if sentence.isdigit():
                raise ValueError('Sentence contains only numbers!')
            if 'lang' in self.kwargs.keys():
                self.lang = self.kwargs.get('lang')
            else:
                self.lang = detect(sentence)
            if self.lang not in self.supported_languages:
                raise ValueError('Language not supported!')
            return sentence
        
        sentence = normalize_space_chars(sentence)
        sentence = normalize_space_seqs(sentence)
        sentence = normalize_apostrophe(sentence)
        sentence = normalize_newline(sentence)
        sentence = validate_sentence(sentence)
        return sentence

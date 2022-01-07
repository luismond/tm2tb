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
# import en_core_web_trf
import en_core_web_sm
# import de_dep_news_trf
# import fr_dep_news_trf

# #from tm2tb import DistanceApi
#model_en = en_core_web_trf.load()
model_en = en_core_web_sm.load()
model_es = es_dep_news_trf.load()
# model_de = de_dep_news_trf.load()
# model_fr = fr_dep_news_trf.load()


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

    def get_pos_tagged_tokens(self):
        """
        Gets a list of tuples [(token, part-of-speech)] from a spAcy doc.
        """
        if self.lang=='en':
            spacy_model = model_en
        if self.lang=='es':
            spacy_model = model_es
        if self.lang=='de':
            spacy_model = model_de
        if self.lang=='fr':
            spacy_model = model_fr

        spacy_doc = spacy_model(self.clean_sentence)
        return [(token.text, token.pos_) for token in spacy_doc]


    def get_pos_tagged_ngrams(self, good_tags = ['NOUN','PROPN'],
                              bad_tags = ['X', 'SCONJ', 'CCONJ', 'AUX'],
                              ngrams_chars_min = 2,
                              ngrams_chars_max = 50,
                              **kwargs):
        """
         Generates a list of ngrams from a list of pos-tagged tokens.
        """
        def get_ngrams(
                       seq,
                       ngrams_min = 1,
                       ngrams_max = 3):
            """
            Generates a list of ngrams from a list, from ngrams_min to ngrams_max:

            print(get_ngrams([a, b, c, d], 1, 3))
            [(a), (b), (c), (d), (a,b), (b,c), (c,d), (a, b, c), (b, c, d)]
            """
            ngrams = [list(zip(*[seq[i:] for i in range(n)]))
                      for n in range(ngrams_min, ngrams_max+1)]
            return [ng for ngl in ngrams for ng in ngl]
        
        ptn = get_ngrams(self.get_pos_tagged_tokens(), **kwargs)

        """
        Filters pos-tagged ngrams.
        """

        # get ngrams longer than ngrams_chars_min
        fptn = list(filter(lambda tl: len(tl[0][0])>ngrams_chars_min, ptn))
        if len(fptn)==0:
            raise ValueError('No ngrams longer than min_ngram_length found!')

        # gets ngrams shorter than ngrams_chars_max
        fptn = list(filter(lambda tl: len(tl[0][0])<ngrams_chars_max, fptn))
        if len(fptn)==0:
            raise ValueError('No ngrams shorter than max_ngram_length found!')

        #keep ngrams with good tags at start and end
        fptn = list(filter(lambda tl: tl[0][1] in good_tags
                          and tl[-1:][0][1] in good_tags, fptn))
        #drop ngrams with punctuation
        fptn = list(filter(lambda tl: tl[0][0].isalpha()
                          and tl[-1:][0][0].isalpha(), fptn))
        # certain puncts not allowed in the middle of the term
        npa = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
               '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        fptn = list(filter(lambda tl:
                          any(t[0] in npa for t in tl) is False, fptn))

        fptn = list(filter(lambda tl:
                          any(t[1] in bad_tags for t in tl) is False, fptn))
        
        
        def get_joined_ngram(tup):
            def rejoin_split_punct(token):
                """
                Joins apostrophes and other special characters to their token.
                """
                def repl(match):
                    groups = match.groups()
                    return '{}{}{}'.format(groups[0],groups[2], groups[3])
                pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
                return re.sub(pattern, repl, token)
            
            jn = ' '.join([t[0] for t in tup])
            jn = rejoin_split_punct(jn)
            return jn
        
        fptn = [(tup, get_joined_ngram(tup)) for tup in fptn]    
        if len(fptn)==0:
            raise ValueError('No pos-tagged_ngrams after filtering!')
                    
        return fptn


    def get_ngrams_to_sentence_distances(self,
                                         server_mode='remote',
                                         diversity=.7,
                                         top_n=35,
                                         **kwargs):
        """
        Sends joined ngrams and sentence to distance server.
        Gets a sorted list of tuples representing ngrams and their distances
        to the sentence.
        """
        ptn = self.get_pos_tagged_ngrams(**kwargs)
        joined_ngrams = [b for (a,b) in ptn]
        seq1 = self.clean_sentence
        seq2 = joined_ngrams
        
        params = json.dumps(
            {'seq1':[seq1],
             'seq2':seq2,
             'diversity':diversity,
             'top_n':top_n,
             'query_type':'ngrams_to_sentence'})

        if server_mode=='remote':
            url = 'http://0.0.0.0:5000/distance_api'
            response = requests.post(url=url, json=params).json()
            best_ngrams = json.loads(response)

        if server_mode=='local':
            best_ngrams = DistanceApi(params).get_top_sentence_ngrams()

        return best_ngrams

    def get_non_overlapping_ngrams(self, **kwargs):
        """
        Takes sorted list of tuples (ngram, distance_to_sentence),
        from closest to farthest and the sentence.
        Returns closest ngrams that do not overlap with farther ngrams.
        Sentence: 'Race to the finish line!'
        Filtered ngrams: [('finish line', 0.1), ('finish', 0.2), ('line', 0.22)]
        'Finish line' is the closest ngram to the sentence.
        We want to avoid having also 'finish' and 'line'.
        """
        nsd = self.get_ngrams_to_sentence_distances(**kwargs)
        sentence = self.clean_sentence
        nsd_new = []
        for tup in nsd:
            ngram = tup[0]
            def repl(match):
                return ' '
            pattern = r"(^|\s|\W)({})($|\s|\W)".format(ngram)
            matches = re.findall(pattern, sentence)
            sentence = re.sub(pattern, repl, sentence)
            if len(matches)==0:
                pass
            else:
                nsd_new.append(tup)
        if len(nsd_new)==0:
            raise ValueError('No ngrams left after removing overlapping ngrams!')
        return nsd_new

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB Sentence class.
Implements methods for string cleaning, validation, tokenization,
ngram generation and ngram selection.

Take a raw string and instantiate a Sentence object:
>>> string = 'Giant pandas in the wild will occasionally eat other grasses,
wild tubers, or even meat in the form of birds, rodents, or carrion.'
>>> sn = Sentence(string)

Inspect the Sentence language:
>>> print(sn.lang)
'en'

Get the sentence tokens:
>>> print(sn.get_tokens())
['Giant', 'pandas', 'in', 'the', 'wild', 'will', 'occasionally', 'eat',
 'other', 'grasses', ',', 'wild', 'tubers', ',', 'or', 'even', 'meat',
 'in', 'the', 'form', 'of', 'birds', ',', 'rodents', ',', 'or', 'carrion', '.']


Get POS-tagged tokens:
>>> print(sn.get_pos_tagged_tokens())
[('Giant', 'ADJ'), ('pandas', 'NOUN'), ('in', 'ADP'), ('the', 'DET'),
 ('wild', 'NOUN'), ('will', 'AUX'), ('occasionally', 'ADV'), ('eat', 'VERB'),
 ('other', 'ADJ'), ('grasses', 'NOUN'), (',', 'PUNCT'), ('wild', 'ADJ'),
 ('tubers', 'NOUN'), (',', 'PUNCT'), ('or', 'CCONJ'), ('even', 'ADV'),
 ('meat', 'NOUN'), ('in', 'ADP'), ('the', 'DET'), ('form', 'NOUN'),
 ('of', 'ADP'), ('birds', 'NOUN'), (',', 'PUNCT'), ('rodents', 'NOUN'),
 (',', 'PUNCT'), ('or', 'CCONJ'), ('carrion', 'NOUN'), ('.', 'PUNCT')]


Get token ngrams (default length: 1-3):
>>> print(sn.get_token_ngrams())
[('Giant',), ('pandas',), ('in',), ('the',)...
 ('Giant', 'pandas'), ('pandas', 'in'), ('in', 'the'), ('the', 'wild')...
 ('Giant', 'pandas', 'in'), ('pandas', 'in', 'the'), ('in', 'the', 'wild')...]


Get n-grams of POS-tagged tokens:
>>> print(sn.get_pos_tagged_ngrams())
[(('Giant', 'ADJ'),), (('pandas', 'NOUN'),), (('in', 'ADP'),)...
 (('Giant', 'ADJ'), ('pandas', 'NOUN')), (('pandas', 'NOUN'), ('in', 'ADP'))...]


Allow tags at the start, middle or end of ngram to filter POS-tagged tokens.
(default: NOUN and PROPN allowed at the start and end):
>>> print(sn.filter_pos_tagged_ngrams())
[(('pandas', 'NOUN'),), (('wild', 'NOUN'),), (('grasses', 'NOUN'),),
 (('tubers', 'NOUN'),), (('meat', 'NOUN'),), (('form', 'NOUN'),),
 (('birds', 'NOUN'),), (('rodents', 'NOUN'),), (('carrion', 'NOUN'),),
 (('form', 'NOUN'), ('of', 'ADP'), ('birds', 'NOUN'))]


Get a list of ngrams sorted by distance to the sentence:
>>> print(sn.get_ngrams_to_sentence_distances())
[('pandas', 1.4132797718048096), ('form of birds', 1.4417308568954468),
 ('birds', 1.4918220043182373), ('wild', 1.5934118032455444),
 ('rodents', 1.596575140953064), ('meat', 1.750525951385498),
 ('form', 1.7685511112213135), ('grasses', 1.775187611579895),
 ('carrion', 1.8042054176330566), ('tubers', 1.9135468006134033)]

Optional: keep only non-overlapping n-grams:
>>> print(sn.get_non_overlapping_ngrams())
[('pandas', 1.4132797718048096), ('form of birds', 1.4417308568954468),
 ('wild', 1.5934118032455444), ('rodents', 1.596575140953064),
 ('meat', 1.750525951385498), ('grasses', 1.775187611579895),
 ('carrion', 1.8042054176330566), ('tubers', 1.9135468006134033)]

"""

import re
import json
from langdetect import detect
import requests

import es_dep_news_trf
import en_core_web_trf
import de_dep_news_trf
import fr_dep_news_trf

model_en = en_core_web_trf.load()
model_es = es_dep_news_trf.load()
model_de = de_dep_news_trf.load()
model_fr = fr_dep_news_trf.load()
spacy_models = {'model_en':model_en,
                'model_es':model_es,
                'model_de':model_de,
                'model_fr':model_fr}

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
        self.sentence_min_length = 40
        self.sentence_max_length = 400
        self.min_non_alpha_ratio = .25
        if 'ngrams_min' in kwargs.keys():
            self.ngrams_min = kwargs.get('ngrams_min')
        else:
            self.ngrams_min = 1
        if 'ngrams_max' in kwargs.keys():
            self.ngrams_max = kwargs.get('ngrams_max')
        else:
            self.ngrams_max = 3
        self.ngrams_chars_min = 2
        self.ngrams_chars_max = 30
        if 'good_tags' in kwargs.keys():
            self.good_tags = kwargs.get('good_tags')
        else:
            self.good_tags = ['NOUN','PROPN']
        if 'bad_tags' in kwargs.keys():
            self.bad_tags = kwargs.get('bad_tags')
        else:
            self.bad_tags = ['X', 'SCONJ', 'CCONJ', 'AUX']

        self.clean_sentence = self.preprocess()

    def preprocess(self):
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
            if non_alpha_ratio >= self.min_non_alpha_ratio:
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
            if len(sentence) <= self.sentence_min_length:
                raise ValueError('Sentence is too short!')
            if len(sentence) >= self.sentence_max_length:
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

    def get_spacy_doc(self):
        """
        Passes a language to instantiate a spAcy object representing a sentence.
        """
        spacy_model = spacy_models['model_{}'.format(self.lang)]
        spacy_doc = spacy_model(self.clean_sentence)
        return spacy_doc

    def get_tokens(self):
        """
        Gets a list of tokens from a spAcy object representing a sentence.
        """
        return [token.text for token in self.get_spacy_doc()]

    def get_pos_tagged_tokens(self):
        """
        Gets a list of tuples [(token, part-of-speech)] from a spAcy doc.
        """
        return [(token.text, token.pos_) for token in self.get_spacy_doc()]

    def get_ngrams(self, seq):
        """
        Generates a list of ngrams from a list, from ngrams_min to ngrams_max:

        print(get_ngrams([a, b, c, d], 1, 3))
        [(a), (b), (c), (d), (a,b), (b,c), (c,d), (a, b, c), (b, c, d)]
        """
        ngrams = [list(zip(*[seq[i:] for i in range(n)]))
                  for n in range(self.ngrams_min, self.ngrams_max+1)]
        return [ng for ngl in ngrams for ng in ngl]

    def get_token_ngrams(self):
        """
        Generates a list of ngrams from a list of spAcy tokens.
        """
        return self.get_ngrams(self.get_tokens())

    def get_pos_tagged_ngrams(self):
        """
         Generates a list of ngrams from a list of pos-tagged tokens.
        """
        return self.get_ngrams(self.get_pos_tagged_tokens())

    def filter_pos_tagged_ngrams(self):
        """
        Filters pos-tagged ngrams.
        """
        ptn = self.get_pos_tagged_ngrams()

        #good_tags = self.good_tags
        #keep ngrams with good tags at start and end
        fptn = list(filter(lambda tl: tl[0][1] in self.good_tags
                          and tl[-1:][0][1] in self.good_tags, ptn))
        #drop ngrams with punctuation
        fptn = list(filter(lambda tl: tl[0][0].isalpha()
                          and tl[-1:][0][0].isalpha(), fptn))
        # certain puncts not allowed in the middle of the term
        npa = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
               '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        fptn = list(filter(lambda tl:
                          any(t[0] in npa for t in tl) is False, fptn))

        fptn = list(filter(lambda tl:
                          any(t[1] in self.bad_tags for t in tl) is False, fptn))
        if len(fptn)==0:
            raise ValueError('No pos-tagged_ngrams after filtering!')
        return fptn

    def get_ngrams_to_sentence_distances(self):
        """
        Joins and validates ngrams.
        Sends joined ngrams and sentence to similarity server.
        Gets a sorted list of tuples representing ngrams and their distances
        to the sentence.
        """

        fptn = self.filter_pos_tagged_ngrams()
        ngrams = [[token for (token, tag) in tuple_list] for tuple_list in fptn]

        def rejoin_split_punct(token):
            """
            Joins apostrophes and other special characters to their token.
            """
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
            return re.sub(pattern, repl, token)

        def validate_ngram_length(joined_ngrams):
            """
            Validates ngram lengths.
            """
            joined_ngrams = list(filter(lambda jn: len(jn)>=2, joined_ngrams))
            if len(joined_ngrams)==0:
                raise ValueError('No ngrams longer than min_ngram_length found!')
            joined_ngrams = list(filter(lambda jn: len(jn)<=30, joined_ngrams))
            if len(joined_ngrams)==0:
                raise ValueError('No ngrams shorter than max_ngram_length found!')
            return joined_ngrams

        joined_ngrams = set(rejoin_split_punct(' '.join(t)) for t in ngrams)
        joined_ngrams = validate_ngram_length(joined_ngrams)

        url = 'http://0.0.0.0:5000/sim_api'
        params = json.dumps(
            {'seq1':joined_ngrams,
            'seq2':[self.clean_sentence]})
        response = requests.post(url=url, json=params).json()
        ngram_to_sentence_distances = json.loads(response)
        ngram_to_sentence_distances = [(a, c) for (a, b, c) in ngram_to_sentence_distances]
        ngram_to_sentence_distances = sorted(ngram_to_sentence_distances, key=lambda t: t[1])
        return ngram_to_sentence_distances

    def get_non_overlapping_ngrams(self):
        """
        Takes sorted list of tuples (ngram, distance_to_sentence),
        from closest to farthest and the sentence.

        Returns closest ngrams that do not overlap with farther ngrams.

        Sentence: 'Race to the finish line!'
        Filtered ngrams: [('finish line', 0.1), ('finish', 0.2), ('line', 0.22)]

        'Finish line' is the closest ngram to the sentence.
        We want to avoid having also 'finish' and 'line'.
        """
        nsd = self.get_ngrams_to_sentence_distances()
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

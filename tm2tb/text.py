#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text class
"""
from collections import Counter as cnt
from random import randint

import pandas as pd
from spacy.tokens import Doc
from langdetect import detect, LangDetectException

from tm2tb.spacy_models import get_spacy_model
from tm2tb import trf_model
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams
pd.options.mode.chained_assignment = None

class Text:
    """
    Implements fast methods for monolingual terminology extraction.
    Text is a list of sentences.
    """
    def __init__(self, text):
        self.text = text
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sentences_clean = self.preprocess_sentences()
        self.lang = self.detect_text_lang()
        self.candidate_ngrams = self.get_candidate_ngrams()

    def preprocess_sentences(self):
        """
        Batch preprocess sentences
        """
        sentences = []
        for sentence in self.text:
            try:
                sentences.append(preprocess(sentence))
            except:
                pass
        if len(sentences)==0:
            raise ValueError('No clean sentences!')
        return sentences

    def detect_text_lang(self):
        """
        Runs langdetect on a sample of the sentences.
        Returns the most commonly detected language.
        """
        sentences = [s for s in self.sentences_clean if len(s)>10]
        sample_len = 20
        if len(sentences)<=sample_len:
            sentences_sample = sentences
        else:
            rand_start = randint(0, (len(sentences)-1)-sample_len)
            sentences_sample = sentences[rand_start:rand_start+sample_len]
        detections = []
        for i in sentences_sample:
            try:
                detections.append(detect(i))
            except LangDetectException:
                pass
        if len(detections)==0:
            raise ValueError('Insufficient data to detect language!')
        lang = cnt(detections).most_common(1)[0][0]

        if lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return lang

    def get_pos_tokens(self):
        """
        Get part-of-speech tags of each token in sentence
        """
        # Get spaCy model
        spacy_model = get_spacy_model(self.lang)
        # Pipe sentences
        sdocs = list(spacy_model.pipe(self.sentences_clean))
        # Concatenate Docs
        sc_doc = Doc.from_docs(sdocs)
        # Extract text and pos from Doc
        pos_tokens = [(token.text, token.pos_) for token in sc_doc]
        return pos_tokens

    def get_pos_ngrams(self, ngrams_min=1, ngrams_max=2):
        """
        Get ngrams from part-of-speech tagged sentences
        """
        pos_tokens = self.get_pos_tokens()
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def get_candidate_ngrams(self,
                             min_freq=1,
                             include_pos=None,
                             exclude_pos=None,
                             **kwargs):
        """
        Get final candidate ngrams from part-of-speech tagged sentences
        """
        pos_ngrams = self.get_pos_ngrams(**kwargs)
        pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=min_freq]
        candidate_ngrams = filter_ngrams(pos_ngrams,
                                         include_pos=include_pos,
                                         exclude_pos=exclude_pos)
        return candidate_ngrams

    def get_embeddings(self):
        embs = trf_model.encode(self.candidate_ngrams['joined_ngrams'])
        return embs

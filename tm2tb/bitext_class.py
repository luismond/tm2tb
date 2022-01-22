#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:46:08 2022

@author: user
"""
from collections import Counter as cnt
from random import randint
import numpy as np
import pandas as pd

from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect

from tm2tb.spacy_models import get_spacy_model
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams

pd.options.mode.chained_assignment = None

print('Loading LaBSE model...')
model = SentenceTransformer('')

class BiText:
    """
    Implements fast methods for bilingual terminology extraction.
    Bitext is a list of tuples of sentences.
    """
    def __init__(self, bitext):
        self.bitext = bitext
        self.src_sentences, self.trg_sentences = zip(*self.bitext)
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sample_len = 20
        self.src_lang = self.detect_text_lang(self.src_sentences)
        self.trg_lang = self.detect_text_lang(self.trg_sentences)

    def detect_text_lang(self, sentences):
        """
        Runs langdetect on a sample of the sentences.
        Returns the most commonly detected language.
        """
        if len(sentences)<=self.sample_len:
            sentences_sample = sentences
        else:
            rand_start = randint(0, (len(sentences)-1)-self.sample_len)
            sentences_sample = sentences[rand_start:rand_start+self.sample_len]
        lang = cnt([detect(i) for i in sentences_sample]).most_common(1)[0][0]
        if lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return lang

    @staticmethod
    def preprocess_sentences(sentences):
        """
        Batch preprocess sentences
        """
        return [preprocess(s) for s in sentences]

    def get_pos_tokens(self, sentences, lang):
        """
        Get part-of-speech tags of each token in sentence
        """
        sentences = self.preprocess_sentences(sentences)
        # Get spaCy model
        spacy_model = get_spacy_model(lang)
        # Pipe sentences
        sdocs = list(spacy_model.pipe(sentences))
        # Concatenate Docs
        sc_doc = Doc.from_docs(sdocs)
        # Extract text and pos from Doc
        pos_tokens = [(token.text, token.pos_) for token in sc_doc]
        return pos_tokens

    def get_pos_ngrams(self, sentences, lang, ngrams_min=1, ngrams_max=2):
        """
        Get ngrams from part-of-speech tagged sentences
        """
        pos_tokens = self.get_pos_tokens(sentences, lang)
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def get_candidate_ngrams(self,
                             sentences,
                             lang,
                             include_pos = None,
                             exclude_pos = None,
                             min_freq=1,
                             **kwargs):
        """
        Get final candidate ngrams from part-of-speech tagged sentences
        """
        pos_ngrams = self.get_pos_ngrams(sentences, lang, **kwargs)
        pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=min_freq]
        candidate_ngrams = filter_ngrams(pos_ngrams, include_pos, exclude_pos)
        return candidate_ngrams

    @staticmethod
    def get_bitext_top_ngrams_partial(ngrams0, ngrams1):
        """
        Get top bitext ngrams from one side
        """
        src_ngrams = ngrams0['joined_ngrams'].tolist()
        trg_ngrams = ngrams1['joined_ngrams'].tolist()
        # Get POS tags
        src_tags = ngrams0['tags'].tolist()
        trg_tags = ngrams1['tags'].tolist()
        # Get embeddings
        src_embeddings = model.encode(src_ngrams)
        trg_embeddings = model.encode(trg_ngrams)
        # Get similarities
        candidate_similarities = cosine_similarity(src_embeddings, trg_embeddings)
        # Get indexes
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
        # Get max trg ngrams values and indexes
        max_trg_values = np.max(candidate_similarities[src_idx][:, trg_idx], axis=1)
        max_trg_idx = np.argmax(candidate_similarities[src_idx][:, trg_idx], axis=1)
        # make ngrams dataframe with the top src_ngram/trg_ngram similarities
        bi_ngrams = pd.DataFrame([(src_ngrams[idx],
                                   src_tags[idx],
                                   trg_ngrams[max_trg_idx[idx]],
                                   trg_tags[max_trg_idx[idx]],
                                   float(max_trg_values[idx])) for idx in src_idx])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']
        # Keep ngrams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= .6]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        # Finish
        bi_ngrams = bi_ngrams.round(4)
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        return bi_ngrams

    def get_top_ngrams_fast(self, **kwargs):
        """
        Extract and filter all source ngrams and all target ngrams.
        Find their most similar matches.
        Much faster, less precise, can cause OOM errors.
        """

        src_ngrams = self.get_candidate_ngrams(self.src_sentences,
                                                self.src_lang,
                                                include_pos = None,
                                                exclude_pos = None,
                                                **kwargs)

        trg_ngrams = self.get_candidate_ngrams(self.trg_sentences,
                                                self.trg_lang,
                                                include_pos = None,
                                                exclude_pos = None,
                                                **kwargs)

        # Get top bitext ngrams from the source side
        bi_ngramss = self.get_bitext_top_ngrams_partial(src_ngrams, trg_ngrams)
        # Get top bitext ngrams from the target side
        bi_ngramst = self.get_bitext_top_ngrams_partial(trg_ngrams, src_ngrams)
        # Rearrange columns
        bi_ngramst.columns = ['trg_ngram',
                             'trg_ngram_tags',
                             'src_ngram',
                             'src_ngram_tags',
                             'bi_ngram_similarity']
        # Concat results
        bi_ngrams = pd.concat([bi_ngramss, bi_ngramst])

        # Drop duplicates
        bi_ngrams['st'] = [''.join(t) for t in list(zip(bi_ngrams['src_ngram'],
                                                        bi_ngrams['trg_ngram']))]
        bi_ngrams = bi_ngrams.drop_duplicates(subset='st')
        bi_ngrams = bi_ngrams.drop(columns=['st'])
        bi_ngrams = bi_ngrams.reset_index()
        # Group by source, get the most similar target n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (src_ngram, df) in list(bi_ngrams.groupby('src_ngram'))])

        # Group by target, get the most similar source n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (trg_ngram, df) in list(bi_ngrams.groupby('trg_ngram'))])
        return bi_ngrams

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import re
from random import randint
from collections import Counter as cnt

from langdetect import detect
import pandas as pd
import numpy as np

from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from tm2tb.preprocess import preprocess
from tm2tb.spacy_models import get_spacy_model

print('Loading LaBSE model...')
model = SentenceTransformer('')

class Bitextr:
    "Implements fast methods for bilingual terminology extraction"
    "Bitext is a list of tuples of sentences"
    def __init__(self, bitext):
        self.bitext = bitext
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sample_len = 20
        self.src_lang, self.trg_lang = self.get_bitext_langs()

    def get_bitext_sample(self):
        rand_start = randint(0,(len(self.bitext)-1)-self.sample_len)
        bitext_sample = self.bitext[rand_start:rand_start+self.sample_len]
        return bitext_sample

    def get_bitext_langs(self):
        if len(self.bitext)<=self.sample_len:
            bitext_sample = self.bitext
        else:
            bitext_sample = self.get_bitext_sample()
        src_lang = detect(' '.join(i[0] for i in bitext_sample))
        trg_lang = detect(' '.join(i[1] for i in bitext_sample))
        if src_lang not in self.supported_languages or\
            trg_lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return src_lang, trg_lang

    def preproc_bitext_sentences(self):
        src_sentences = []
        trg_sentences = []
        for i, _ in enumerate(self.bitext):
            try:
                src_sentence_raw = self.bitext[i][0]
                src_sentence = preprocess(src_sentence_raw)
                src_sentences.append(src_sentence)
                trg_sentence_raw = self.bitext[i][1]
                trg_sentence = preprocess(trg_sentence_raw)
                trg_sentences.append(trg_sentence)
            except:
                pass
        if len(src_sentences) == 0 or len(trg_sentences) == 0:
            raise ValueError('No clean sentences left!')
        return src_sentences, trg_sentences

    def get_bitext_pos_tagged_tokens(self):
        src_sentences, trg_sentences = self.preproc_bitext_sentences()
        src_model = get_spacy_model(self.src_lang)
        sdocs = list(src_model.pipe(src_sentences))
        sc_doc = Doc.from_docs(sdocs)
        src_pos_tagged_tokens = [(token.text, token.pos_) for token in sc_doc]
        trg_model = get_spacy_model(self.trg_lang)
        tdocs = list(trg_model.pipe(trg_sentences))
        tc_doc = Doc.from_docs(tdocs, ensure_whitespace=True)
        trg_pos_tagged_tokens = [(token.text, token.pos_) for token in tc_doc]
        return src_pos_tagged_tokens, trg_pos_tagged_tokens

    @staticmethod
    def get_ngrams(pos_tokens):
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(1, 4+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def generate_bitext_ngrams(self):
        src_pos_tagged_tokens, trg_pos_tagged_tokens = self.get_bitext_pos_tagged_tokens()
        src_pos_tagged_ngrams = self.get_ngrams(src_pos_tagged_tokens)
        trg_pos_tagged_ngrams = self.get_ngrams(trg_pos_tagged_tokens)
        return src_pos_tagged_ngrams, trg_pos_tagged_ngrams

    @staticmethod
    def get_candidate_ngrams(pos_ngrams, include_pos = None, exclude_pos = None):
        freq_min = 2
        pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=freq_min]
        #trg_pos_tagged_ngrams = [a for a,b in cnt(list(trg_pos_tagged_ngrams)).items() if b>=freq_min]
        exclude_punct = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
                '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']
        if include_pos is None:
            include_pos = ['NOUN', 'PROPN', 'ADJ']
        if exclude_pos is None:
            exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX', 'VERB']
            exclude_pos = [tag for tag in exclude_pos if not tag in include_pos]
        # Keep ngrams where the first element's pos-tag
        # and the last element's pos-tag are present in include_pos
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][1] in include_pos
                          and pos_ngram[-1:][0][1] in include_pos, pos_ngrams)
        # Keep ngrams where none of elements' tag is in exclude pos
        pos_ngrams = filter(lambda pos_ngram: not any(token[1] in exclude_pos
                                                      for token in pos_ngram), pos_ngrams)
        # Keep ngrams where the first element's token
        # and the last element's token are alpha
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][0].isalpha()
                          and pos_ngram[-1:][0][0].isalpha(), pos_ngrams)
        # Keep ngrams where none of the middle elements' text is in exclude punct
        pos_ngrams = filter(lambda pos_ngram: not any((token[0] in exclude_punct
                                                       for token in pos_ngram[1:-1])), pos_ngrams)

        # check if POS n-grams are empty
        pos_ngrams = [list(pn) for pn in pos_ngrams]
        if len(pos_ngrams)==0:
            raise ValueError('No POS n-grams left after filtering!')

        def rejoin_special_punct(ngram):
            'Joins apostrophes and other special characters to their token.'
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
            return re.sub(pattern, repl, ngram)

        # Make data frame from n-grams and parts-of-speech
        pos_ngrams_ = pd.DataFrame([zip(*pos_ngram) for pos_ngram in pos_ngrams])
        pos_ngrams_.columns = ['ngrams','tags']
        pos_ngrams_.loc[:, 'joined_ngrams'] = \
            pos_ngrams_['ngrams'].apply(lambda ng: rejoin_special_punct(' '.join(ng)))
        pos_ngrams_ = pos_ngrams_.drop_duplicates(subset='joined_ngrams')
        pos_ngrams_ = pos_ngrams_.reset_index()
        pos_ngrams_ = pos_ngrams_.drop(columns=['index'])
        return pos_ngrams_

    def get_bitext_candidate_ngrams(self):
        src_pos_tagged_ngrams, trg_pos_tagged_ngrams = self.generate_bitext_ngrams()
        src_candidate_ngrams = self.get_candidate_ngrams(src_pos_tagged_ngrams)
        trg_candidate_ngrams = self.get_candidate_ngrams(trg_pos_tagged_ngrams)
        return src_candidate_ngrams, trg_candidate_ngrams

    @staticmethod
    def get_bitext_top_ngrams_partial(ngrams0, ngrams1):
        # Get top bitext ngrams from one side
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
        # # Keep ngrams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= .85]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        # Finish
        bi_ngrams = bi_ngrams.round(4)
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        # print('finnish')
        # print(time()-START)
        return bi_ngrams

    def get_bitext_top_ngrams(self):
        src_ngrams, trg_ngrams = self.get_bitext_candidate_ngrams()
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
        return bi_ngrams

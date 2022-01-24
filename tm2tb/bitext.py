#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiText class
"""
from typing import Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from tm2tb import BitextReader
from tm2tb import Text

class BiText:
    """
    Implements fast methods for bilingual terminology extraction.
    """
    def __init__(self, input_: Union[str, Tuple[tuple]]):

        if isinstance(input_, str):
            #If input is a bitext path, read bitext, instantiate each column as Text
            bitext = BitextReader(input_).read_bitext()
            self.src_text = Text(bitext['src'].tolist())
            self.trg_text = Text(bitext['trg'].tolist())

        if isinstance(input_, tuple):
            #If input is two text paths, read texts, instantiate each text as Text
            src_text_path = input_[0]
            trg_text_path = input_[1]
            self.src_text = Text(self.read_text(src_text_path))
            self.trg_text = Text(self.read_text(trg_text_path))

        self.src_ngrams_df = self.src_text.get_candidate_ngrams()
        self.trg_ngrams_df = self.trg_text.get_candidate_ngrams()
        self.src_embs = self.src_text.get_embeddings()
        self.trg_embs = self.trg_text.get_embeddings()

    @staticmethod
    def read_text(text_path):
        with open(text_path, 'r', encoding='utf8') as file:
            en_text = file.read().split('\n')
        return en_text

    @staticmethod
    def get_seq_similarities(src_embeddings, trg_embeddings):
        # Get source/target ngram similarity matrix
        src_seq_similarities = cosine_similarity(src_embeddings,
                                              trg_embeddings)

        # Get target/source ngram similarity matrix
        trg_seq_similarities = cosine_similarity(trg_embeddings,
                                              src_embeddings)
        return src_seq_similarities, trg_seq_similarities

    def get_aligned_ngrams(self):
        """
        Get top bitext ngrams from one side
        """
        src_ngrams = self.src_ngrams_df['joined_ngrams'].tolist()
        trg_ngrams = self.trg_ngrams_df['joined_ngrams'].tolist()
        # Get POS tags
        src_tags = self.src_ngrams_df['tags'].tolist()
        trg_tags = self.trg_ngrams_df['tags'].tolist()
        # Get embeddings
        src_embeddings = self.src_embs
        trg_embeddings = self.trg_embs

        # Get similarities
        src_seq_similarities, trg_seq_similarities = self.get_seq_similarities(src_embeddings,
                                                                               trg_embeddings)
        # Get indexes
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))

        # Get indexes and values of most similar source ngram for each target ngram
        src_max_values = np.max(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        src_max_idx = np.argmax(trg_seq_similarities[trg_idx][:, src_idx], axis=1)

        # Get indexes and values of most similar target ngram for each source ngram
        trg_max_values = np.max(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        trg_max_idx = np.argmax(src_seq_similarities[src_idx][:, trg_idx], axis=1)

        # make ngrams dataframe with the top src_ngram/trg_ngram similarities
        src_aligned_ngrams = pd.DataFrame([(src_ngrams[idx],
                                            src_tags[idx],
                                            trg_ngrams[trg_max_idx[idx]],
                                            trg_tags[trg_max_idx[idx]],
                                            float(trg_max_values[idx])) for idx in src_idx])

        # make ngrams dataframe with the top trg_ngram/src_ngram similarities
        trg_aligned_ngrams = pd.DataFrame([(src_ngrams[src_max_idx[idx]],
                                            src_tags[src_max_idx[idx]],
                                            trg_ngrams[idx],
                                            trg_tags[idx],
                                            float(src_max_values[idx])) for idx in trg_idx])

        return src_aligned_ngrams, trg_aligned_ngrams

    def get_top_ngrams(self, min_similarity=.8, **kwargs):
        """
        Extract and filter all source ngrams and all target ngrams.
        Find their most similar matches.
        Much faster, less precise, can cause OOM errors.
        """
        src_aligned_ngrams, trg_aligned_ngrams = self.get_aligned_ngrams(**kwargs)
        bi_ngrams = pd.concat([src_aligned_ngrams, trg_aligned_ngrams])
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']
        # Keep n-grams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= min_similarity]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        # For one-word terms, keep those longer than 1 character
        bi_ngrams = bi_ngrams[bi_ngrams['src_ngram'].str.len()>1]
        bi_ngrams = bi_ngrams[bi_ngrams['trg_ngram'].str.len()>1]
        # Group by source, get the most similar target n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (src_ngram, df) in list(bi_ngrams.groupby('src_ngram'))])
        # Group by target, get the most similar source n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (trg_ngram, df) in list(bi_ngrams.groupby('trg_ngram'))])
        bi_ngrams = bi_ngrams.round(4)
        return bi_ngrams

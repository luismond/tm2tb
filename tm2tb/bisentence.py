#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiSentence class. Inherits from Sentence
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tm2tb import Sentence
pd.options.mode.chained_assignment = None

class BiSentence:
    def __init__(self, sentence_tuple):
        self.src_sentence, self.trg_sentence = sentence_tuple

    def get_src_trg_top_ngrams(self, **kwargs):
        # Get source ngram dataframe
        src_ngrams_df = Sentence(self.src_sentence).get_top_ngrams(return_embs=True, **kwargs)
        # Get target ngram dataframe
        trg_ngrams_df = Sentence(self.trg_sentence).get_top_ngrams(return_embs=True, **kwargs)
        return src_ngrams_df, trg_ngrams_df

    @staticmethod
    def get_seq_similarities(src_ngrams_df, trg_ngrams_df):
        # Get source/target ngram similarity matrix
        src_seq_similarities = cosine_similarity(src_ngrams_df['embedding'].tolist(),
                                              trg_ngrams_df['embedding'].tolist())
        # Get target/source ngram similarity matrix
        trg_seq_similarities = cosine_similarity(trg_ngrams_df['embedding'].tolist(),
                                              src_ngrams_df['embedding'].tolist())
        return src_seq_similarities, trg_seq_similarities

    def get_aligned_ngrams(self, **kwargs):
        # Get ngrams, pos_tags and ranks from source & target sentences
        src_ngrams_df, trg_ngrams_df = self.get_src_trg_top_ngrams(**kwargs)
        src_seq_similarities, trg_seq_similarities = self.get_seq_similarities(src_ngrams_df,
                                                                               trg_ngrams_df)
        src_ngrams = src_ngrams_df['joined_ngrams'].tolist()
        trg_ngrams = trg_ngrams_df['joined_ngrams'].tolist()
        src_tags = src_ngrams_df['tags'].tolist()
        trg_tags = trg_ngrams_df['tags'].tolist()
        src_ranks = src_ngrams_df['rank'].tolist()
        trg_ranks = trg_ngrams_df['rank'].tolist()
        # Get source ngram & target ngram indexes
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
        # Get indexes and values of most similar target ngram for each source ngram
        src_max_values = np.max(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        src_max_idx = np.argmax(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        # Get indexes and values of most similar source ngram for each target ngram
        trg_max_values = np.max(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        trg_max_idx = np.argmax(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        # Align target ngrams & metadata with source ngrams & metadata
        src_aligned_ngrams = pd.DataFrame([(src_ngrams[idx],
                                            src_ranks[idx],
                                            src_tags[idx],
                                            trg_ngrams[trg_max_idx[idx]],
                                            trg_ranks[trg_max_idx[idx]],
                                            trg_tags[trg_max_idx[idx]],
                                            float(trg_max_values[idx])) for idx in src_idx])
        # Align source ngrams & metadata with target ngrams & metadata
        trg_aligned_ngrams = pd.DataFrame([(src_ngrams[src_max_idx[idx]],
                                            src_ranks[src_max_idx[idx]],
                                            src_tags[src_max_idx[idx]],
                                            trg_ngrams[idx],
                                            trg_ranks[idx],
                                            trg_tags[idx],
                                            float(src_max_values[idx])) for idx in trg_idx])
        return src_aligned_ngrams, trg_aligned_ngrams

    def get_top_ngrams(self, min_similarity=.8, **kwargs):
        # Concatenate source & target ngram alignments
        src_aligned_ngrams, trg_aligned_ngrams = self.get_aligned_ngrams(**kwargs)
        bi_ngrams = pd.concat([src_aligned_ngrams, trg_aligned_ngrams])
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_rank',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_rank',
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
        # Get bi n-gram rank
        bi_ngrams['bi_ngram_rank'] = bi_ngrams['bi_ngram_similarity'] * \
            bi_ngrams['src_ngram_rank'] * bi_ngrams['trg_ngram_rank']
        bi_ngrams = bi_ngrams.sort_values(by='bi_ngram_rank', ascending=False)
        bi_ngrams = bi_ngrams.round(4)
        return bi_ngrams

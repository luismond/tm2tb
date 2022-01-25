#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rank ngrams
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_ngrams(cand_ngrams_df,
                joined_ngrams,
                seq1_embeddings,
                seq2_embeddings,
                top_n=100,
                diversity=.8,
                return_embs=False):
    
    # Get sentence/ngrams similarities
    ngram_sentence_sims = cosine_similarity(seq2_embeddings, seq1_embeddings)
    
    # Get ngrams/ngrams similarities
    ngram_sims = cosine_similarity(seq2_embeddings)
    
    # Initialize candidates and choose best ngram
    best_ngrams_idx = [np.argmax(ngram_sentence_sims)]
    
    # All ngrams that are not in best ngrams
    candidates_idx = [i for i in range(len(joined_ngrams)) if i != best_ngrams_idx[0]]
    
    for _ in range(min(top_n - 1, len(joined_ngrams) - 1)):
        # Get distances within candidates and between candidates and selected ngrams
        candidate_sims = ngram_sentence_sims[candidates_idx, :]
        rest_ngrams_sims = np.max(ngram_sims[candidates_idx][:, best_ngrams_idx], axis=1)
    
        # Calculate Maximum Marginal Relevance
        mmr = (1-diversity) * candidate_sims - diversity * rest_ngrams_sims.reshape(-1, 1)
    
        # Get closest candidate
        mmr_idx = candidates_idx[np.argmax(mmr)]
    
        # Update best ngrams & candidates
        best_ngrams_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
    
    # Keep only ngrams in best_ngrams_idx
    best_ngrams_df = cand_ngrams_df.iloc[best_ngrams_idx]
    
    # Add rank and embeddings
    best_ngrams_df.loc[:, 'rank'] = [round(float(ngram_sentence_sims.reshape(1, -1)[0][idx]), 4)
                                for idx in best_ngrams_idx]
    best_ngrams_df.loc[:, 'embedding'] = [seq2_embeddings[idx] for idx in best_ngrams_idx]
    best_ngrams_df = best_ngrams_df.sort_values(by='rank', ascending = False)
    # Keep ngrams with rank above 0
    best_ngrams_df = best_ngrams_df[best_ngrams_df['rank']>0.01]
    if return_embs is False:
        best_ngrams_df = best_ngrams_df.drop(columns=['embedding'])
    return best_ngrams_df

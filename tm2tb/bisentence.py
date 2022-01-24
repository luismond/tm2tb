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

    def get_ngrams_dfs(self, **kwargs):
        # Get source ngram dataframe
        src_ngrams_df = Sentence(self.src_sentence).get_top_ngrams(return_embs=True, **kwargs)
        # Get target ngram dataframe
        trg_ngrams_df = Sentence(self.trg_sentence).get_top_ngrams(return_embs=True, **kwargs)
        return src_ngrams_df, trg_ngrams_df

    @staticmethod
    def get_seq_similarities(src_embs, trg_embs):
        seq_similarities = cosine_similarity(src_embs, trg_embs)
        return seq_similarities

    def get_aligned_ngrams(self, **kwargs):
        src_ngrams_df, trg_ngrams_df = self.get_ngrams_dfs(**kwargs)
        src_ngrams = src_ngrams_df['joined_ngrams'].tolist()
        src_tags = src_ngrams_df['tags'].tolist()
        src_ranks = src_ngrams_df['rank'].tolist()
        src_embeddings = src_ngrams_df['embedding'].tolist()
        trg_ngrams = trg_ngrams_df['joined_ngrams'].tolist()
        trg_tags = trg_ngrams_df['tags'].tolist()
        trg_ranks = trg_ngrams_df['rank'].tolist()
        trg_embeddings = trg_ngrams_df['embedding'].tolist()
        seq_similarities = self.get_seq_similarities(src_embeddings, trg_embeddings)
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
        # Get indexes and values of most similar source ngram for each target ngram
        trg_max_values = np.max(seq_similarities[src_idx][:, trg_idx], axis=1)
        trg_max_idx = np.argmax(seq_similarities[src_idx][:, trg_idx], axis=1)
        # Get indexes and values of most similar target ngram for each source ngram
        src_max_values = np.max(seq_similarities[src_idx][:, trg_idx], axis=0)
        src_max_idx = np.argmax(seq_similarities[src_idx][:, trg_idx], axis=0)
        src_aligned_ngrams = pd.DataFrame([(src_ngrams[idx],
                                            src_tags[idx],
                                            src_ranks[idx],
                                            trg_ngrams[trg_max_idx[idx]],
                                            trg_tags[trg_max_idx[idx]],
                                            trg_ranks[trg_max_idx[idx]],
                                            float(trg_max_values[idx])) for idx in src_idx])
        trg_aligned_ngrams = pd.DataFrame([(src_ngrams[src_max_idx[idx]],
                                            src_tags[src_max_idx[idx]],
                                            src_ranks[src_max_idx[idx]],
                                            trg_ngrams[idx],
                                            trg_tags[idx],
                                            trg_ranks[idx],
                                            float(src_max_values[idx])) for idx in trg_idx])
        return src_aligned_ngrams, trg_aligned_ngrams

    def get_top_ngrams(self, min_similarity=.8, **kwargs):
        # Concatenate source & target ngram alignments
        src_aligned_ngrams, trg_aligned_ngrams = self.get_aligned_ngrams(**kwargs)
        bi_ngrams = pd.concat([src_aligned_ngrams, trg_aligned_ngrams])
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_tags',
                             'src_ngram_rank',
                             'trg_ngram',
                             'trg_ngram_tags',
                             'trg_ngram_rank',
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

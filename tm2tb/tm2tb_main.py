"""
tm2tb main
"""
#import json
#import faiss

from typing import Union, Tuple, List
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from tm2tb import Sentence
from tm2tb import BilingualReader

MODEL_PATH = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_server/labse_model'
model = SentenceTransformer(MODEL_PATH)
#%%
class Tm2Tb:
    """
    main class to do all stuff
    """
    def __init__(self, model=model):
        self.model = model

    def get_ngrams(self,
                   input_: Union[str, Tuple[tuple], List[list]],
                   **kwargs):

        if isinstance(input_, str):     # get ngrams from one sentence
            ngrams = self.get_sentence_ngrams(input_, **kwargs)

        elif isinstance(input_, tuple): # get bilingual ngrams from two sentences
            src_sentence, trg_sentence = input_
            ngrams = self.get_bi_ngrams_from_bisentence(src_sentence,
                                                        trg_sentence,
                                                        **kwargs)
        elif isinstance(input_, list): # get bilingual ngrams from a bitext
            bitext = input_
            ngrams = self.get_bi_ngrams_from_bitext(bitext,
                                                    **kwargs)

        return ngrams

    def get_sentence_ngrams(self, sentence,
                            top_n = None,
                            diversity=.5,
                            return_embs=False,
                            **kwargs):
        sent = Sentence(sentence)
        clean_sentence = sent.clean_sentence
        ngrams = sent.get_ngrams(**kwargs)
        ngrams = list(set(ngrams['joined_ngrams']))

        def get_best_ngrams(ngrams, top_n):
            seq1_embeddings = self.model.encode([clean_sentence])
            seq2_embeddings = self.model.encode(ngrams)
            ngram_sentence_dists = cosine_similarity(seq2_embeddings,
                                                     seq1_embeddings)
            ngram_dists = cosine_similarity(seq2_embeddings)
            # Initialize candidates and choose best ngram
            best_ngrams_idx = [np.argmax(ngram_sentence_dists)]
            # All ngrams that are not in best ngrams
            candidates_idx = [i for i in range(len(ngrams)) if i != best_ngrams_idx[0]]
            if top_n is None:
                top_n = round(len(ngrams)*.5)
            for _ in range(min(top_n - 1, len(ngrams) - 1)):
                # Get distances within candidates and between candidates and selected ngrams
                candidate_sims = ngram_sentence_dists[candidates_idx, :]
                rest_ngrams_sims = np.max(ngram_dists[candidates_idx][:, best_ngrams_idx], axis=1)
                # Calculate MMR
                mmr = (1-diversity) * candidate_sims - diversity * rest_ngrams_sims.reshape(-1, 1)
                # Get closest candidate
                mmr_idx = candidates_idx[np.argmax(mmr)]
                # Update best ngrams & candidates
                best_ngrams_idx.append(mmr_idx)
                candidates_idx.remove(mmr_idx)
                # Get best ngrams
                best_ngrams = [(ngrams[idx],
                                round(float(ngram_sentence_dists.reshape(1, -1)[0][idx]), 4),
                                seq2_embeddings[idx])
                                for idx in best_ngrams_idx]
            return best_ngrams

        best_ngrams = get_best_ngrams(ngrams, top_n)
        best_ngrams = sorted(best_ngrams, key=lambda tup: tup[1], reverse=True)

        if return_embs is False:
            best_ngrams = [(ngram, ngram_sent_sim)
                           for (ngram, ngram_sent_sim, ngram_emb) in best_ngrams]
        return best_ngrams

    def get_bi_ngrams_from_bisentence(self,
                                      src_sentence,
                                      trg_sentence,
                                      min_similarity=.75,
                                      **kwargs):

        src_ngrams, src_ngv, seq1_embeddings = zip(*self.get_sentence_ngrams(src_sentence,
                                                                             return_embs=True,
                                                                             **kwargs))

        trg_ngrams, trg_ngv, seq2_embeddings = zip(*self.get_sentence_ngrams(trg_sentence,
                                                                             return_embs=True,
                                                                             **kwargs))

        def get_max_seq_similarities(seq1_embeddings, seq2_embeddings):
            seq_similarities = cosine_similarity(seq1_embeddings,
                                                 seq2_embeddings)
            # get seq1 & seq2 indexes
            seq1_idx = list(range(len(src_ngrams)))
            seq2_idx = list(range(len(trg_ngrams)))

            # get max seq2 values and indexes
            max_seq2_values = np.max(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
            max_seq2_idx = np.argmax(seq_similarities[seq1_idx][:, seq2_idx], axis=1)

            # get max seq similarities
            max_seq_similarities = [(src_ngrams[idx],
                                     src_ngv[idx],
                                     trg_ngrams[max_seq2_idx[idx]],
                                     trg_ngv[max_seq2_idx[idx]],
                                     float(round(max_seq2_values[idx], 4))) for idx in seq1_idx]
            return max_seq_similarities

        max_seq_similarities = get_max_seq_similarities(seq1_embeddings, seq2_embeddings)
        #Make bi_ngrams dataframe
        bi_ngrams = pd.DataFrame(max_seq_similarities)
        bi_ngrams.columns = ['src', 'src_s', 'trg', 'trg_s', 'similarity']

        bi_ngrams = self._filter_bi_ngrams(bi_ngrams, min_similarity)
        return bi_ngrams

    def get_bi_ngrams_from_bitext(self, bitext, min_similarity=.75, **kwargs):
        bi_ngrams = []
        for i, _ in enumerate(bitext):
            try:
                src_sentence = bitext[i][0]
                trg_sentence = bitext[i][1]
                bi_ngrams_ = self.get_bi_ngrams_from_bisentence(src_sentence,
                                                                trg_sentence,
                                                                **kwargs)
                for bi_ngram in bi_ngrams_:
                    bi_ngrams.append(bi_ngram)
            except ValueError:
                pass

        bi_ngrams = pd.DataFrame(bi_ngrams)
        bi_ngrams.columns = ['src', 'src_s', 'trg', 'trg_s', 'similarity', 'x']
        bi_ngrams = self._filter_bi_ngrams(bi_ngrams, min_similarity, get_ngram_sent_avgs=True)
        return bi_ngrams

    @classmethod
    def read_bitext(cls, file_path):
        path, file_name = os.path.split(file_path)
        bitext = BilingualReader(path, file_name).get_bitext()
        bitext = list(zip(bitext['src'],bitext['trg']))
        return bitext

    @classmethod
    def _filter_bi_ngrams(cls, bi_ngrams,
                          min_similarity,
                          get_ngram_sent_avgs=False):

        # Keep ngrams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['similarity'] >= min_similarity]

        def group_and_get_best_match(bi_ngrams):
            # Group by source, get most similar target ngram
            bi_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                                for (src_ngram, df) in list(bi_ngrams.groupby('src'))])

            # Group by target, get most similar source ngram
            bi_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                                for (trg_ngram, df) in list(bi_ngrams.groupby('trg'))])
            return bi_ngrams

        def get_ngram_to_sentence_avgs(bi_ngrams):
            #Get ngram to sentence averages from all extracted ngrams
            src_avg = {a: round(b['src_s'].sum()/len(b), 4)
                       for (a, b) in list(bi_ngrams.groupby('src'))}
            trg_avg = {a: round(b['trg_s'].sum()/len(b), 4)
                       for (a, b) in list(bi_ngrams.groupby('trg'))}
            bi_ngrams = bi_ngrams.drop_duplicates(subset='src')
            bi_ngrams = bi_ngrams.drop_duplicates(subset='trg')
            bi_ngrams['src_s'] = bi_ngrams['src'].apply(lambda ngram: src_avg[ngram])
            bi_ngrams['trg_s'] = bi_ngrams['trg'].apply(lambda ngram: trg_avg[ngram])
            return bi_ngrams

        if get_ngram_sent_avgs is True:
            bi_ngrams = get_ngram_to_sentence_avgs(bi_ngrams)

        bi_ngrams = group_and_get_best_match(bi_ngrams)
        bi_ngrams['x'] = bi_ngrams['similarity']*bi_ngrams['src_s']*bi_ngrams['trg_s']
        bi_ngrams = bi_ngrams.sort_values(by='x', ascending=False)

        # Turn to list
        bi_ngrams = list(zip(bi_ngrams['src'],
                              bi_ngrams['src_s'],
                              bi_ngrams['trg'],
                              bi_ngrams['trg_s'],
                              bi_ngrams['similarity'],
                              bi_ngrams['x']))

        bi_ngrams = [(a, b, c, d, round(e,4), round(f,4))
                     for (a,b,c,d,e,f) in bi_ngrams]

        return bi_ngrams

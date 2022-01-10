"""
tm2tb main
"""
#import json
import os
#import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tm2tb import Sentence
from tm2tb import BilingualReader
from sentence_transformers import SentenceTransformer
MODEL_PATH = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_server/labse_model'
model = SentenceTransformer(MODEL_PATH)
from typing import Union, Tuple

class Tm2Tb:
    """
    main class to do all stuff
    """
    def __init__(self, model=model):
        self.model = model

    @classmethod
    def get_sentence_ngrams(self, sentence,
                            top_n = None,
                            diversity=.5,
                            ngrams_min=None,
                            ngrams_max=None,
                            include_pos=None,
                            exclude_pos=None):
        """
        Get top_n most similar ngrams from a sentence.

        Parameters
        ----------
        sentence : str
            Raw Unicode sentence, short text or paragraph.
        top_n : int, optional
            DESCRIPTION. Number of best ngrams to return.
                         If None, top_n = number of total ngrams * .5
        diversity : int, optional
            DESCRIPTION. Diversity value for Maximal Marginal Relevance. The default is .8.
        ngrams_min : int, optional
            DESCRIPTION. Minimum ngram sequence length.
        ngrams_max : int, optional
            DESCRIPTION. Maximum ngram sequence length.
        include_pos : List, optional
            DESCRIPTION. A list of POS-tags to delimit the ngrams.
        exclude_pos : List, optional
            DESCRIPTION. A list of POS-tags to exclude from the ngrams.

        Returns
        -------
        List
            DESCRIPTION. A list of tuples (ngram, similarity_value).

            [('panda', 0.4116),
             ('bear', 0.2271),
             ('diet', 0.1889),
             ('wild', 0.1726),
             ('rodents', 0.1718),
             ('fish', 0.144)]

        """
        sn = Sentence(sentence)
        ngrams = sn.get_ngrams(ngrams_min=ngrams_min,
                               ngrams_max=ngrams_max,
                               include_pos=include_pos,
                               exclude_pos=exclude_pos)

        #sentence = sn.clean_sentence
        ngrams = list(set(ngrams['joined_ngrams']))

        seq1_embeddings = model.encode([sn.clean_sentence])
        seq2_embeddings = model.encode(ngrams)

        ngram_sentence_dists = cosine_similarity(seq2_embeddings, seq1_embeddings)
        ngram_dists = cosine_similarity(seq2_embeddings)

        # Initialize candidates and choose best ngram
        best_ngrams_idx = [np.argmax(ngram_sentence_dists)]

        # All ngrams that are not in best ngrams
        candidates_idx = [i for i in range(len(ngrams)) if i != best_ngrams_idx[0]]

        if top_n is None:
            top_n = round(len(ngrams)*.5)

        for _ in range(min(top_n - 1, len(ngrams) - 1)):
            # Get distances within candidates and between candidates and selected ngrams
            candidate_similarities = ngram_sentence_dists[candidates_idx, :]

            target_similarities = np.max(ngram_dists[candidates_idx][:, best_ngrams_idx], axis=1)

            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)

            # Get closest candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update best ngrams & candidates
            best_ngrams_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        # Get best ngrams
        top_sentence_ngrams = [(ngrams[idx],
                                round(float(ngram_sentence_dists.reshape(1, -1)[0][idx]), 4))
                               for idx in best_ngrams_idx]

        return sorted(top_sentence_ngrams, key=lambda tup: tup[1], reverse=True)

    @classmethod
    def get_max_sim(self, src_ngrams, trg_ngrams):
        seq1_embeddings = model.encode(src_ngrams)
        seq2_embeddings = model.encode(trg_ngrams)
        seq_similarities = cosine_similarity(seq1_embeddings,
                                             seq2_embeddings)
        # get seq1 & seq2 indexes
        seq1_idx = list(range(len(src_ngrams))) #[i for i in range(len(src_ngrams))]
        seq2_idx = list(range(len(trg_ngrams)))

        # get max seq2 values and indexes
        max_seq2_values = np.max(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
        max_seq2_idx = np.argmax(seq_similarities[seq1_idx][:, seq2_idx], axis=1)

        # get max seq similarities
        max_seq_similarities = [(src_ngrams[idx], trg_ngrams[max_seq2_idx[idx]],
                                  float(round(max_seq2_values[idx], 4))) for idx in seq1_idx]

        # sort max seq similarities
        max_seq_similarities = sorted(max_seq_similarities, key=lambda tup: tup[2], reverse=True)
        return max_seq_similarities

    def get_bi_ngrams(self,
                      docs: Union[str, Tuple[tuple]],
                      min_similarity=.75,
                      **kwargs):

        if isinstance(docs, tuple):
            src_sentence, trg_sentence = docs
            bi_ngrams = self.get_bi_ngrams_from_bisentence(src_sentence,
                                                                  trg_sentence,
                                                                  **kwargs)
        elif isinstance(docs, str):
            file_path = docs
            bi_ngrams = self.get_bi_ngrams_from_bitext(file_path,
                                                       **kwargs)

        # Group by source, get closest target ngram
        bi_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (src_ngram, df) in list(bi_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bi_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (trg_ngram, df) in list(bi_ngrams.groupby('trg'))])

        # Filter by similarity
        bi_ngrams = bi_ngrams[bi_ngrams['similarity'] >= min_similarity]
        bi_ngrams['x'] = bi_ngrams['similarity']*bi_ngrams['src_s']*bi_ngrams['trg_s']
        bi_ngrams = bi_ngrams.sort_values(by='x', ascending=False)

        # Turn to list
        # bi_ngrams = list(zip(bi_ngrams['src'],bi_ngrams['trg'],
        #                             bi_ngrams['similarity'],bi_ngrams['src_s'],
        #                             bi_ngrams['trg_s']))
        # bi_ngrams = [(a, b, round(c, 4), d, e) for (a,b,c,d,e) in bi_ngrams]
        return bi_ngrams

    def get_bi_ngrams_from_bisentence(self,
                              src_sentence,
                              trg_sentence,
                              **kwargs):

        src_ngrams, src_ngrams_to_sent_sims = zip(*self.get_sentence_ngrams(src_sentence,
                                                       **kwargs))
        trg_ngrams, trg_ngrams_to_sent_sims = zip(*self.get_sentence_ngrams(trg_sentence,
                                                       **kwargs))

        sd = dict(zip(src_ngrams,src_ngrams_to_sent_sims))
        td = dict(zip(trg_ngrams,trg_ngrams_to_sent_sims))

        max_seq_similarities = self.get_max_sim(src_ngrams, trg_ngrams)

        #Make bi_ngrams dataframe
        bi_ngrams = pd.DataFrame(max_seq_similarities)
        bi_ngrams.columns = ['src', 'trg', 'similarity']

        bi_ngrams['src_s'] = bi_ngrams['src'].apply(lambda x: sd[x])
        bi_ngrams['trg_s'] = bi_ngrams['trg'].apply(lambda x: td[x])
        return bi_ngrams


    def get_bi_ngrams_from_bitext(self, file_path, bn_mode, **kwargs):
        path, file_name = os.path.split(file_path)
        bitext = BilingualReader(path, file_name).get_bitext()

        if bn_mode=='iterative':
            all_bi_ngrams = []
            for i in range(len(bitext)):
                try:
                    src_sentence = bitext.iloc[i]['src']
                    trg_sentence = bitext.iloc[i]['trg']
                    bi_ngrams = self.get_bi_ngrams_from_bisentence(src_sentence,
                                                                    trg_sentence,
                                                                    **kwargs)
                    all_bi_ngrams.append(bi_ngrams)
                except:
                    pass

            # Make bi_ngrams dataframe
            bi_ngrams = pd.concat(all_bi_ngrams)
            bi_ngrams = bi_ngrams.reset_index()
            bi_ngrams = bi_ngrams.drop(columns=['index'])

        if bn_mode=='fast':
            all_sng = []
            all_tng = []
            for i in range(len(bitext)):
                try:
                    src_sentence = bitext.iloc[i]['src']
                    trg_sentence = bitext.iloc[i]['trg']
                    src_ngrams = self.get_sentence_ngrams(src_sentence,
                                                          **kwargs)
                    for ng in src_ngrams:
                        if not ng in all_sng:
                            all_sng.append(ng)

                    trg_ngrams = self.get_sentence_ngrams(trg_sentence,
                                                           **kwargs)
                    for ng in trg_ngrams:
                        if not ng in all_tng:
                            all_tng.append(ng)
                except:
                    pass
            src_ngrams, src_ngrams_to_sent_sims = zip(*all_sng)
            trg_ngrams, trg_ngrams_to_sent_sims = zip(*all_tng)

            sd = dict(zip(src_ngrams,src_ngrams_to_sent_sims))
            td = dict(zip(trg_ngrams,trg_ngrams_to_sent_sims))

            max_seq_similarities = self.get_max_sim(src_ngrams, trg_ngrams)

            #Make bi_ngrams dataframe
            bi_ngrams = pd.DataFrame(max_seq_similarities)
            bi_ngrams.columns = ['src', 'trg', 'similarity']

            bi_ngrams['src_s'] = bi_ngrams['src'].apply(lambda x: sd[x])
            bi_ngrams['trg_s'] = bi_ngrams['trg'].apply(lambda x: td[x])

        return bi_ngrams

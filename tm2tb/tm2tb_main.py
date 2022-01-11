from typing import Union, Tuple, List
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from tm2tb import Sentence
from tm2tb import BilingualReader

model = SentenceTransformer('LabSE')

class Tm2Tb:
    """
    A class to represent a term/ngram/keyword extractor.

    Attributes
    ----------
    model: sentence_transformers.SentenceTransformer.SentenceTransformer
        LaBSE sentence transformer model.
        https://huggingface.co/sentence-transformers/LaBSE

    Methods
    -------
    get_ngrams()
        Gets best ngrams/terms/keywords from sentences, tuples of sentences or list of tuples of sentences.
    """
    def __init__(self, model=model):
        self.model = model

    def get_ngrams(self, input_: Union[str, Tuple[tuple], List[list]], **kwargs):
        """
        Parameters
        ----------
        input_ : Union[str, Tuple[tuple], List[list]]
            str:    A string representing a sentence or short paragraph.

            tuple:  A tuple of two sentences or short paragraphs.
                    (For example, a sentence in English and its translation to Spanish).

            list:   A list of tuples of two sentences.

        **kwargs : dict
            See below

        Optional Keyword Arguments:
            ngrams_min : int, optional
                DESCRIPTION. Minimum ngram sequence length.
                             The default value is 1.

            ngrams_max : int, optional
                DESCRIPTION. Maximum ngram sequence length.
                             The default value is 2.

            include_pos : List, optional
                DESCRIPTION. A list of POS-tags to delimit the ngrams.
                            If None, the default value is ['NOUN', 'PROPN']

            exclude_pos : List, optional
                DESCRIPTION. A list of POS-tags to exclude from the ngrams.

            top_n : int, optional
                DESCRIPTION. An integer representing the maximum number of results
                             of single sentence ngrams.
                             The default value is len(candidate_ngrams)*.5

            diversity : float, optional
                DESCRIPTION.    A float representing the diversity of single sentence results.
                                It is used to calculate the Maximal Marginal Relevance.
                                The default value is 0.5

            min_similarity : float, optional
                DESCRIPTION. A float representing the minimum similarity between
                             ngrams from sentence A and ngrams from sentence B.
                             The similarity is obtained by calculating the
                             cosine similarity of the LaBSE-embedded ngrams.
                             The default value is .75

        Returns
        -------
        ngrams : TYPE
            DESCRIPTION.

        """
        if isinstance(input_, str):
            ngrams = self._get_best_sentence_ngrams(input_, **kwargs)

        elif isinstance(input_, tuple):
            src_sentence, trg_sentence = input_
            ngrams = self._get_bi_ngrams_from_bisentence(src_sentence,
                                                        trg_sentence,
                                                        **kwargs)
        elif isinstance(input_, list):
            bitext = input_
            ngrams = self._get_bi_ngrams_from_bitext(bitext,
                                                    **kwargs)

        return ngrams

    def _get_best_sentence_ngrams(self,
                                  sentence,
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

    def _get_bi_ngrams_from_bisentence(self,
                                       src_sentence,
                                       trg_sentence,
                                       min_similarity=.75,
                                       **kwargs):

        # Get sentence ngrams, ngrams-to-sentence similarities and embeddings
        src_ngrams_ = self._get_best_sentence_ngrams(src_sentence,
                                                     return_embs=True,
                                                     **kwargs)

        trg_ngrams_ = self._get_best_sentence_ngrams(trg_sentence,
                                                     return_embs=True,
                                                     **kwargs)

        # Get best similarities between source ngrams and target ngrams
        def get_max_seq_similarities(src_ngrams_, trg_ngrams_):

            src_ngrams, src_ngrams_sent_sims, src_ngrams_embs = zip(*src_ngrams_)
            trg_ngrams, trg_ngrams_sent_sims, trg_ngrams_embs = zip(*trg_ngrams_)

            seq_similarities = cosine_similarity(src_ngrams_embs,
                                                 trg_ngrams_embs)
            # get seq1 & seq2 indexes
            seq1_idx = list(range(len(src_ngrams)))
            seq2_idx = list(range(len(trg_ngrams)))

            # get max seq2 values and indexes
            max_seq2_values = np.max(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
            max_seq2_idx = np.argmax(seq_similarities[seq1_idx][:, seq2_idx], axis=1)

            # get max seq similarities
            max_seq_similarities = [(src_ngrams[idx],
                                     src_ngrams_sent_sims[idx],
                                     trg_ngrams[max_seq2_idx[idx]],
                                     trg_ngrams_sent_sims[max_seq2_idx[idx]],
                                     float(round(max_seq2_values[idx], 4))) for idx in seq1_idx]
            return max_seq_similarities

        max_seq_similarities = get_max_seq_similarities(src_ngrams_, trg_ngrams_)

        # Make bi_ngrams dataframe
        bi_ngrams = pd.DataFrame(max_seq_similarities)
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_sent_sim',
                             'trg_ngram',
                             'trg_ngram_sent_sim',
                             'bi_ngram_similarity']

        # Filter bi_ngrams
        bi_ngrams = self._filter_bi_ngrams(bi_ngrams, min_similarity)
        return bi_ngrams

    def _get_bi_ngrams_from_bitext(self, bitext, min_similarity=.75, **kwargs):
        bi_ngrams = []
        for i, _ in enumerate(bitext):
            try:
                src_sentence = bitext[i][0]
                trg_sentence = bitext[i][1]
                bi_ngrams_ = self._get_bi_ngrams_from_bisentence(src_sentence,
                                                                trg_sentence,
                                                                **kwargs)
                for bi_ngram in bi_ngrams_:
                    bi_ngrams.append(bi_ngram)
            except ValueError:
                pass

        bi_ngrams = pd.DataFrame(bi_ngrams)
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_sent_sim',
                             'trg_ngram',
                             'trg_ngram_sent_sim',
                             'bi_ngram_similarity',
                             'relevance']

        bi_ngrams = self._filter_bi_ngrams(bi_ngrams, min_similarity, get_ngram_sent_avgs=True)
        return bi_ngrams

    @classmethod
    def read_bitext(cls, file_path):
        """
        Parameters
        ----------
        file_path : str
            String representing a full path to a bilingual file
            See tm2tb.bilingual_reader.BilingualReader.

        Returns
        -------
        bitext : List
            DESCRIPTION. A list of tuples of sentences.

        """
        path, file_name = os.path.split(file_path)
        bitext = BilingualReader(path, file_name).get_bitext()
        bitext = list(zip(bitext['src'],bitext['trg']))
        return bitext

    @classmethod
    def _filter_bi_ngrams(cls, bi_ngrams,
                          min_similarity,
                          get_ngram_sent_avgs=False):

        # Keep ngrams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= min_similarity]

        def group_and_get_best_match(bi_ngrams):
            # Group by source, get most similar target ngram
            bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                                for (src_ngram, df) in list(bi_ngrams.groupby('src_ngram'))])

            # Group by target, get most similar source ngram
            bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                                for (trg_ngram, df) in list(bi_ngrams.groupby('trg_ngram'))])
            return bi_ngrams

        def get_ngram_to_sentence_avgs(bi_ngrams):
            #Get ngram to sentence averages from all extracted ngrams
            src_avg = {a: round(b['src_ngram_sent_sim'].sum()/len(b), 4)
                       for (a, b) in list(bi_ngrams.groupby('src_ngram'))}
            trg_avg = {a: round(b['trg_ngram_sent_sim'].sum()/len(b), 4)
                       for (a, b) in list(bi_ngrams.groupby('trg_ngram'))}
            bi_ngrams = bi_ngrams.drop_duplicates(subset='src_ngram')
            bi_ngrams = bi_ngrams.drop_duplicates(subset='trg_ngram')
            bi_ngrams['src_ngram_sent_sim'] = bi_ngrams['src_ngram'].apply(lambda ngram: src_avg[ngram])
            bi_ngrams['trg_ngram_sent_sim'] = bi_ngrams['trg_ngram'].apply(lambda ngram: trg_avg[ngram])
            return bi_ngrams

        def get_relevance(bi_ngrams):
            # Multiply source ngram-to-sentence similarity,
            # target ngram-to-sentence similarity
            # and ngram-to-ngram similarity.
            bi_ngrams['relevance'] = bi_ngrams['bi_ngram_similarity'] * \
                bi_ngrams['src_ngram_sent_sim'] * bi_ngrams['trg_ngram_sent_sim']
            bi_ngrams = bi_ngrams.sort_values(by='relevance', ascending=False)
            return bi_ngrams
        
        if get_ngram_sent_avgs is True:
            bi_ngrams = get_ngram_to_sentence_avgs(bi_ngrams)
        bi_ngrams = group_and_get_best_match(bi_ngrams)
        bi_ngrams = get_relevance(bi_ngrams)
        
        # Turn to list
        bi_ngrams = list(zip(bi_ngrams['src_ngram'],
                              bi_ngrams['src_ngram_sent_sim'],
                              bi_ngrams['trg_ngram'],
                              bi_ngrams['trg_ngram_sent_sim'],
                              bi_ngrams['bi_ngram_similarity'],
                              bi_ngrams['relevance']))

        bi_ngrams = [(a, b, c, d, round(e,4), round(f,4))
                     for (a, b, c, d, e, f) in bi_ngrams]
        
        return bi_ngrams

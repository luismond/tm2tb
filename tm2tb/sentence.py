import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

from tm2tb.spacy_models import get_spacy_model
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams
from tm2tb import trf_model
pd.options.mode.chained_assignment = None

class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence
        self.supported_languages = ['en', 'es']#, 'de', 'fr']
        self.clean_sentence = preprocess(self.sentence)
        self.lang = self.validate_lang()

    def validate_lang(self):
        lang = detect(self.clean_sentence)
        if lang not in self.supported_languages:
            raise ValueError('Language not supported!')
        return lang
        # else:
        #     return lang

    def _generate_ngrams(self, ngrams_min = 1, ngrams_max = 2):
        """
        Generate ngrams from sentence sequence
        """
        # Get spaCy model and instantiate a doc with the clean sentence
        spacy_model = get_spacy_model(self.lang)
        doc = spacy_model(self.clean_sentence)
        # Get text and part-of-speech tag for each token in document
        pos_tokens = [(token.text, token.pos_) for token in doc]
        # Get n-grams from pos_tokens
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def _get_candidate_ngrams(self, include_pos = None, exclude_pos = None, **kwargs):
        pos_ngrams = self._generate_ngrams(**kwargs)
        pos_ngrams = filter_ngrams(pos_ngrams, include_pos, exclude_pos)
        return pos_ngrams

    def get_top_ngrams(self, top_n = None, diversity=.8, return_embs=False, **kwargs):
        """
        Embed sentence and candidate ngrams.
        Calculate the best sentence ngrams using cosine similarity and MMR.
        """
        cand_ngrams_df = self._get_candidate_ngrams(**kwargs)
        joined_ngrams = cand_ngrams_df['joined_ngrams']

        if top_n is None:
            top_n = round(len(joined_ngrams)*.85)

        # Embed clean sentence and joined ngrams
        seq1_embeddings = trf_model.encode([self.clean_sentence])
        seq2_embeddings = trf_model.encode(joined_ngrams)

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
        if return_embs is False:
            best_ngrams_df = best_ngrams_df.drop(columns=['ngrams','tags','embedding'])
        return best_ngrams_df

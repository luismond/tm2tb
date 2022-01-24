"""
Sentence class
"""
import pandas as pd
from langdetect import detect
from tm2tb.spacy_models import get_spacy_model
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams
from tm2tb.rank_ngrams import rank_ngrams
from tm2tb import trf_model
pd.options.mode.chained_assignment = None

class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence
        self.supported_languages = ['en', 'es']
        self.clean_sentence = preprocess(self.sentence)
        self.lang = self.validate_lang()

    def validate_lang(self):
        lang = detect(self.clean_sentence)
        if lang not in self.supported_languages:
            raise ValueError('Language not supported!')
        return lang

    def get_pos_tokens(self):
        spacy_model = get_spacy_model(self.lang)
        doc = spacy_model(self.clean_sentence)
        pos_tokens = [(token.text, token.pos_) for token in doc]
        return pos_tokens

    def get_candidate_ngrams(self, **kwargs):
        pos_tokens = self.get_pos_tokens()
        candidate_ngrams = filter_ngrams(pos_tokens, **kwargs)
        return candidate_ngrams

    def get_top_ngrams(self, top_n=None, diversity=.8, **kwargs):
        """
        Embed sentence and candidate ngrams.
        Calculate the best sentence ngrams using cosine similarity and MMR.
        """
        cand_ngrams_df = self.get_candidate_ngrams(**kwargs)
        joined_ngrams = cand_ngrams_df['joined_ngrams']

        if top_n is None:
            top_n = round(len(joined_ngrams)*.85)

        seq1_embeddings = trf_model.encode([self.clean_sentence])
        seq2_embeddings = trf_model.encode(joined_ngrams)

        # Rank ngrams
        top_ngrams = rank_ngrams(cand_ngrams_df,
                                 joined_ngrams,
                                 seq1_embeddings,
                                 seq2_embeddings,
                                 top_n=top_n,
                                 diversity=diversity)
        return top_ngrams

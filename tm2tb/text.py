"""
Text class
"""
from collections import Counter as cnt
from random import randint
import pandas as pd
from spacy.tokens import Doc
from langdetect import detect, LangDetectException

from tm2tb import trf_model
from tm2tb.spacy_models import get_spacy_model
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams
from tm2tb.rank_ngrams import rank_ngrams

pd.options.mode.chained_assignment = None

class Text:
    """
    Class representing a Text object
    Contains the text sentences
    Detects the text language
    Implements a fast spaCy method for tokenizing and POS-tagging
    Generates term candidates from the text
    """
    def __init__(self, text):
        self.text = text
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sentences_clean = self.preprocess_sentences()
        self.lang = self.detect_text_lang()

    def preprocess_sentences(self):
        """
        Batch preprocess sentences
        """
        sentences = []
        for sentence in self.text:
            try:
                sentences.append(preprocess(sentence))
            except ValueError:
                pass
        if len(sentences)==0:
            raise ValueError('No clean sentences!')
        return sentences

    def detect_text_lang(self):
        """
        Runs langdetect on a sample of the sentences.
        Returns the most commonly detected language.
        """
        sentences = [s for s in self.sentences_clean if len(s)>10]
        sample_len = 20
        if len(sentences)<=sample_len:
            sentences_sample = sentences
        else:
            rand_start = randint(0, (len(sentences)-1)-sample_len)
            sentences_sample = sentences[rand_start:rand_start+sample_len]
        detections = []
        for i in sentences_sample:
            try:
                detections.append(detect(i))
            except LangDetectException:
                pass
        if len(detections)==0:
            raise ValueError('Insufficient data to detect language!')
        lang = cnt(detections).most_common(1)[0][0]

        if lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return lang

    def get_pos_tokens(self):
        """
        Get part-of-speech tags of each token in sentence
        """
        # Get spaCy model
        spacy_model = get_spacy_model(self.lang)
        # Pipe sentences
        sdocs = list(spacy_model.pipe(self.sentences_clean))
        # Concatenate Docs
        sc_doc = Doc.from_docs(sdocs)
        # Extract text and pos from Doc
        pos_tokens = [(token.text, token.pos_) for token in sc_doc]
        return pos_tokens

    def get_candidate_ngrams(self, **kwargs):
        pos_tokens = self.get_pos_tokens()
        candidate_ngrams = filter_ngrams(pos_tokens, **kwargs)
        return candidate_ngrams

    def get_top_ngrams(self, top_n = None, diversity=.8, return_embs=False, **kwargs):
        """
        Embed sentence and candidate ngrams.
        Calculate the best sentence ngrams using cosine similarity and MMR.
        """
        cand_ngrams_df = self.get_candidate_ngrams(**kwargs)
        joined_ngrams = cand_ngrams_df['joined_ngrams']
        
        if top_n is None:
            top_n = round(len(joined_ngrams)*.85)
        
        # Embed document and joined ngrams
        sent_embs = trf_model.encode(self.sentences_clean)
        doc_embedding = sum(sent_embs)/len(sent_embs)
        doc_embedding = doc_embedding.reshape(1, -1)
        seq2_embeddings = trf_model.encode(joined_ngrams)
        
        top_ngrams = rank_ngrams(cand_ngrams_df,
                                 joined_ngrams,
                                 doc_embedding,
                                 seq2_embeddings,
                                 top_n=100,
                                 diversity=diversity,
                                 return_embs=return_embs)
        
        return top_ngrams

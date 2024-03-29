"""
Extract terms from a sentence or multiple sentences.

Classes
-------
    TermExtractor

Methods
-------
    extract_terms()

"""

from collections import defaultdict
from collections import ChainMap
from typing import Union
from functools import cached_property
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Token, Span
from tm2tb import trf_model
from tm2tb import get_spacy_model
from tm2tb.utils import detect_lang


class TermExtractor:
    """
    Class representing a monolingual term extractor.

    Attributes
    ----------
    input_ : Union[str, list]
        String or list of strings
    lang: str
        Optional two-character language identifier

    Methods
    -------
    extract_terms(self, **kwargs)

    """

    def __init__(self, input_: Union[str, list], lang=None):
        if isinstance(input_, str):
            self.input_ = [input_]
        if isinstance(input_, list):
            self.input_ = input_
        if lang is None:
            self.lang = detect_lang(self.input_)
        else:
            self.lang = lang
        self.docs = list(self.spacy_model.pipe(self.input_))
        self.emb_dims = trf_model.get_sentence_embedding_dimension()

        # Register additional span attributes
        Span.set_extension("similarity", default=None, force=True)
        Span.set_extension("stop_similarity", default=None, force=True)
        Span.set_extension("rank", default=None, force=True)
        Span.set_extension("span_id", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        Span.set_extension("frequency", default=None, force=True)
        Span.set_extension("docs_idx", default=None, force=True)
        Span.set_extension("true_case", default=None, force=True)
        Token.set_extension("true_case", default=None, force=True)


    @cached_property
    def spacy_model(self):
        return get_spacy_model(self.lang)

    def extract_terms(self,
                      return_as_table=True,
                      span_range=(1, 2),
                      freq_min=1,
                      max_stopword_similarity=.55,
                      collapse_similarity=True,
                      incl_pos=None,
                      excl_pos=None):
        """
        Extract terms from one sentence or multiple sentences.

        Parameters
        ----------
        return_as_table : bool, optional
            Return the results as pandas dataframe, the default is True

        span_range : tuple, optional
            Length range of the terms. The default is (1, 2)

        freq_min : int, optional
            Minimum ocurrence frequency of the terms. The default is 1

        incl_pos : list, optional
            List of part-of-speech tags to be included. See `_set_span_extensions`

        excl_pos : list, optional
            List of part-of-speech tags to be excluded. See `_set_span_extensions`

        Returns
        -------
        terms : list of spaCy spans or a Pandas dataframe
            A collection representing the extracted terms
        """
        # Spans
        self._set_span_rules(incl_pos, excl_pos)
        spans_dicts = self._get_spans_dicts(span_range, freq_min)
        spans_freqs_dict = spans_dicts.maps[0]
        spans_docs_dict = spans_dicts.maps[1]
        spans_texts_dict = spans_dicts.maps[2]

        # Embeddings & similarities
        docs_embeddings_avg = self._get_docs_embeddings_avg(spans_dicts)
        spans_embeddings = trf_model.encode(list(spans_texts_dict.keys()))
        spans_doc_sims = cosine_similarity(spans_embeddings, docs_embeddings_avg)
        stops_embeddings_avg = np.load(f'stops_vectors/{self.emb_dims}/{self.lang}.npy')
        spans_stops_sims = cosine_similarity(spans_embeddings, stops_embeddings_avg)

        # Add metadata to the spans
        top_spans = []
        for idx, emb in enumerate(spans_embeddings):
            span = list(spans_texts_dict.values())[idx]
            similarity = round(float(spans_doc_sims.reshape(1, -1)[0][idx]), 4)
            stop_similarity = round(float(spans_stops_sims.reshape(1, -1)[0][idx]), 4)
            if stop_similarity <= max_stopword_similarity:
                span._.similarity = similarity
                span._.frequency = spans_freqs_dict[span._.true_case]
                span._.span_id = idx
                span._.docs_idx = spans_docs_dict[span._.true_case]
                span._.embedding = emb
                top_spans.append(span)

        top_spans = self._rank_spans(top_spans)

        if collapse_similarity is True:
            top_spans = self._collapse_similarity(top_spans)

        for idx, span in enumerate(top_spans):
            span._.span_id = idx
        top_spans = sorted(top_spans, key=lambda span: span._.span_id)

        if return_as_table is True:
            top_spans = self._return_as_table(top_spans)
        return top_spans

    @staticmethod
    def _rank_spans(spans):
        """
        Calculate the terms ranking.

        1. Normalize similarity and frequency to values between 0 and 1
        2. Sum normalized similarity and frequency
        3. Normalize the resulting sums to values between 0 and 1 (1 being the top-ranked term)
        """
        spans_ = []

        similarities = [s._.similarity for s in spans]
        sims_norm = [sim/max(similarities) for sim in similarities]

        frequencies = [s._.frequency for s in spans]
        freqs_norm = [freq/max(frequencies) for freq in frequencies]

        ranks = [(a+b) for a, b in zip(sims_norm, freqs_norm)]
        ranks_norm = [rank/max(ranks) for rank in ranks]

        for n, span in enumerate(spans):
            span._.rank = ranks_norm[n]
            spans_.append(span)
        return spans_

    @staticmethod
    def _set_span_rules(incl_pos=None, excl_pos=None):
        """
        Set custom attributes and properties on the spans.

        The attributes are used to filter the list of candidate spans.

        Parameters
        ----------
         incl_pos : list, optional
             List of POS tags that must exist at the edges of the span.
             If None, the default is ['NOUN', 'PROPN', 'ADJ']

         excl_pos : list, optional
             List of part-of-speech tags to be excluded from the terms.
             If None, the default is all tags that are not in incl_pos

        Returns
        -------
        None
            Returns nothing.

        """
        pos_tags = [
            'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
            'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE'
            ]
        if incl_pos is None or len(incl_pos) == 0:
            incl_pos = ['NOUN', 'PROPN', 'ADJ']
        if excl_pos is None or len(excl_pos) == 0:
            excl_pos = [tag for tag in pos_tags if tag not in incl_pos]

        # Define span rules
        def _incl_pos(span):
            return span[0].pos_ in incl_pos and span[-1].pos_ in incl_pos

        def _excl_pos(span):
            return not any(t.pos_ in excl_pos for t in span)

        def _alpha_edges(span):
            return span[0].text.isalpha() and span[-1].text.isalpha()

        # Register span rules
        Span.set_extension("incl_pos_edges", getter=_incl_pos, force=True)
        Span.set_extension("excl_pos_any", getter=_excl_pos, force=True)
        Span.set_extension("alpha_edges", getter=_alpha_edges, force=True)

    def _get_spans_dicts(self, span_range, freq_min):
        """Collect spans and add them to span dicts."""
        spans_freqs_dict = defaultdict(int)  # All spans and their frequencies
        spans_docs_dict = defaultdict(set)   # All spans and the documents in which they occur
        spans_texts_dict = defaultdict(set)  # All spans and their string representation

        for doc in self.docs:
            spans = []
            for i in range(len(doc)):
                for n in range(span_range[0], span_range[1]+1):
                    if i+n <= len(doc):
                        span = doc[i:i+n]
                        spans.append(span)

            for span in spans:
                if all((span._.incl_pos_edges, span._.excl_pos_any, span._.alpha_edges)):
                    if len(span.text) > 1:
                        for tok in span:
                            if tok.pos_ == 'PROPN':
                                tok._.true_case = tok.text
                            else:
                                if tok.is_upper is True:
                                    tok._.true_case = tok.text
                                else:
                                    tok._.true_case = tok.text.lower()
                        span._.true_case = ''.join(
                            [''.join((tok._.true_case, tok.whitespace_)) for tok in span]).strip()
                        spans_freqs_dict[span._.true_case] += 1
                        spans_texts_dict[span._.true_case] = span
                        spans_docs_dict[span._.true_case].add(self.docs.index(span.doc))

        for span, freq in spans_freqs_dict.items():
            if freq < freq_min:
                spans_docs_dict.pop(span)
                spans_texts_dict.pop(span)
        if len(spans_texts_dict) == 0:
            raise ValueError(f"No terms left with frequency {freq_min}")

        spans_dicts = ChainMap(spans_freqs_dict, spans_docs_dict, spans_texts_dict)
        return spans_dicts

    def _get_docs_embeddings_avg(self, spans_dicts):
        """Get the documents' average embedding."""
        top_docs_idx = set(itertools.chain(*list(spans_dicts.maps[1].values())))
        top_docs_texts = list(self.docs[i].text for i in top_docs_idx)
        docs_embeddings = trf_model.encode([text for text in top_docs_texts if len(text) > 0])
        docs_embeddings_avg = sum(docs_embeddings)/len(docs_embeddings)
        return docs_embeddings_avg.reshape(1, -1)

    @staticmethod
    def _collapse_similarity(spans):
        """
        Given candidates with differing casing, keep only the highest ranked one.

        For example, "panda bear", "Panda bear" and "Panda Bear"

        """
        # Sort spans by rank
        spans = sorted(spans, key=lambda span: span._.similarity, reverse=True)
        # Take their saved vectors
        span_embs = [s._.embedding for s in spans]
        # Make terms similarity matrix
        sm = cosine_similarity(span_embs, span_embs)
        # Use the terms' true case representation
        spans_texts = [sp._.true_case for sp in spans]
        seen_values = set()
        # For each idx in axis 0
        index = list(range(sm.shape[0]))
        for idx in index:
            key = spans_texts[idx]
            if key not in seen_values:
                # Take the sim values of all other terms except itself's
                row = sm[:, idx:]
                values_idx = [i for i in index if row[i][0] > .9 and spans_texts[i] != key]
                values = [spans_texts[i] for i in values_idx]
                if len(values) > 0:
                    # For each of these top values
                    for i in values_idx:
                        seen_values.add(spans_texts[i])
                        # Remove them from index
                        index.remove(i)
        return [spans[i] for i in index]

    @staticmethod
    def _return_as_table(spans):
        """Take a list spaCy spans, return it as pandas dataframe."""
        terms = []
        for span in spans:
            text = span._.true_case
            tags = [t.pos_ for t in span]
            freq = span._.frequency
            rank = span._.rank
            terms.append((text, tags, rank, freq))
        terms = pd.DataFrame(terms)
        terms.columns = ['term', 'pos_tags', 'rank', 'frequency']
        terms = terms.sort_values(by='rank', ascending=False)
        terms.reset_index(drop=True, inplace=True)
        terms = terms.round(4)
        return terms
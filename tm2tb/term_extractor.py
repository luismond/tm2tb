"""
Extract terms from a sentence or multiple sentences.

Classes:
    TermExtractor

Functions:
    extract_terms(Union[str, list])
"""

from collections import defaultdict
from collections import namedtuple
from collections import ChainMap
from typing import Union
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Span
from tm2tb import trf_model
from tm2tb import get_spacy_model
from tm2tb.utils import detect_lang, preprocess


class TermExtractor:
    """
    Class representing a simple term/keyword extractor.

    Attributes
    ----------
    input_ : Union[str, list]
        String or list of strings

    Methods
    -------
    extract_terms(self, return_as_tuples=bool, **kwargs)

    """

    def __init__(self, input_: Union[str, list]):
        if isinstance(input_, str):
            self.input_ = [input_]
        if isinstance(input_, list):
            self.input_ = input_
        self.lang = detect_lang(self.input_)
        self.spacy_model = get_spacy_model(self.lang)
        self.docs = list(self.spacy_model.pipe(self.input_))

        # Register additional span attributes
        Span.set_extension("similarity", default=None, force=True)
        Span.set_extension("rank", default=None, force=True)
        Span.set_extension("span_id", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        Span.set_extension("frequency", default=None, force=True)
        Span.set_extension("docs_idx", default=None, force=True)

    def extract_terms(self,
                      return_as_table=True,
                      span_range=(1, 2),
                      freq_min=1,
                      diversity=.9,
                      incl_pos=None,
                      excl_pos=None):
        """
        Extract terms from one sentence or multiple sentences.

        Parameters
        ----------
        return_as_tuples : bool, optional
            Return the results as named tuples. The default is True.

        span_range : tuple, optional
            Length range of the terms. The default is (1, 3).

        freq_min : int, optional
            Minimum ocurrence frequency of the terms. The default is 1.

        incl_pos : list, optional
            See `_set_span_extensions`

        excl_pos : list, optional
            See `_set_span_extensions`

        Returns
        -------
        terms : list of named tuples
            A list of tuples representing the best terms from the input.
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

        # Build spans
        top_spans = []
        for ix in range(len(spans_embeddings)):
            span = list(spans_texts_dict.values())[ix]
            similarity = round(float(spans_doc_sims.reshape(1, -1)[0][ix]), 4)
            span._.similarity = similarity
            span._.frequency = spans_freqs_dict[span.text]
            span._.rank = span._.similarity
            span._.span_id = ix
            span._.docs_idx = spans_docs_dict[span.text]
            span._.embedding = spans_embeddings[ix]
            top_spans.append(span)

        top_spans = sorted(top_spans, key=lambda span: span._.span_id)

        if return_as_table is True:
            top_spans = self._return_as_table(top_spans)
        return top_spans

    @staticmethod
    def _set_span_rules(incl_pos=None, excl_pos=None):
        """
        Set custom attributes and properties on the document Spans.

        The attributes are used to filter the list of candidate spans.

        Tags are simple UPOS part-of-speech tags.

        Parameters
        ----------
         incl_pos : list, optional
             List of POS tags that must be present at the edges of the span.
             If None, the default is ['NOUN', 'PROPN', 'ADJ']

         excl_pos : list, optional
             List of part-of-speech tags to be excluded from the terms.
             If None, the default is all tags that are not in incl_pos

        Returns
        -------
        None
            Returns nothing.

        """
        pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET',
                    'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                    'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
        if incl_pos is None or len(incl_pos) == 0:
            incl_pos = ['NOUN', 'PROPN', 'ADJ']
        if excl_pos is None or len(excl_pos) == 0:
            excl_pos = [tag for tag in pos_tags if tag not in incl_pos]

        # Define span rules
        def incl_pos_(span):
            return span[0].pos_ in incl_pos and span[-1].pos_ in incl_pos

        def excl_pos_(span):
            return not any(t.pos_ in excl_pos for t in span)

        def alpha_edges(span):
            return span[0].text.isalpha() and span[-1].text.isalpha()

        # Register span rules
        Span.set_extension("incl_pos_edges", getter=incl_pos_, force=True)
        Span.set_extension("excl_pos_any", getter=excl_pos_, force=True)
        Span.set_extension("alpha_edges", getter=alpha_edges, force=True)

    def _get_spans_dicts(self, span_range, freq_min):
        """Collect spans and add them to span dicts."""
        spans_freqs_dict = defaultdict(int) #All spans and their frequencies.
        spans_docs_dict = defaultdict(set) #All spans and the documents in which they occur.
        spans_texts_dict = defaultdict(set) #All spans and their string representation.

        for doc in self.docs:
            span_ranges = list((i, i+n) for i in range(len(doc)) \
                               for n in range(span_range[0], span_range[1]+1))
            spans = (doc[n:n_] for (n, n_) in span_ranges)
            for span in spans:
                if span._.incl_pos_edges is True and span._.excl_pos_any is True \
                    and span._.alpha_edges is True and len(span.text) > 1:
                        spans_freqs_dict[span.text] += 1
                        spans_texts_dict[span.text] = span
                        spans_docs_dict[span.text].add(self.docs.index(span.doc))

        for span, freq in spans_freqs_dict.items():
            if freq < freq_min:
                spans_docs_dict.pop(span)
                spans_texts_dict.pop(span)
        if len(spans_texts_dict) == 0:
            raise ValueError(f"No terms left with frequency {freq_min}")

        spans_dicts = ChainMap(spans_freqs_dict, spans_docs_dict, spans_texts_dict)
        return spans_dicts

    def _get_docs_embeddings_avg(self, spans_dicts):
        top_docs_idx = set(itertools.chain(*list(spans_dicts.maps[1].values())))
        top_docs_texts = list(self.docs[i].text for i in top_docs_idx)
        docs_embeddings = trf_model.encode([text for text in top_docs_texts if len(text) > 0])
        docs_embeddings_avg = sum(docs_embeddings)/len(docs_embeddings)
        return docs_embeddings_avg.reshape(1, -1)

    @staticmethod
    def _return_as_table(spans):
        """Take a list spaCy spans, return it as pandas dataframe."""
        terms = []
        for span in spans:
            text = span.text
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

    @staticmethod
    def _mmr_rank(spans_embeddings, spans_doc_sims, diversity, spans_dicts):
        spans_freqs_dict = spans_dicts.maps[0]
        spans_docs_dict = spans_dicts.maps[1]
        spans_texts_dict = spans_dicts.maps[2]
        # Construct a similarity matrix of spans
        spans_sims = cosine_similarity(spans_embeddings)
        top_spans = []
        top_n = len(spans_embeddings)
        # Choose best span to initialize the best spans index
        best_spans_idx = [np.argmax(spans_doc_sims)]
        # Initialize the candidates index
        candidates_idx = [i for i in range(len(spans_embeddings)) if i != best_spans_idx[0]]
        # Iteratively, select the best span, add it to best_spans_idx and remove it from candidates_idx
        for ix in range(min(top_n - 1, len(spans_embeddings) - 1)):
            candidate_sims = spans_doc_sims[candidates_idx, :]
            rest_spans_sims = np.max(spans_sims[candidates_idx][:, best_spans_idx], axis=1)
            # Calculate Maximum Marginal Relevance
            mmr = (1-diversity) * candidate_sims - diversity * rest_spans_sims.reshape(-1, 1)
            # Get best candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]
            # Update best spans & candidates
            best_spans_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

            span = list(spans_texts_dict.values())[mmr_idx]
            similarity = round(float(spans_doc_sims.reshape(1, -1)[0][mmr_idx]), 4)
            span._.similarity = similarity
            span._.frequency = spans_freqs_dict[span.text]
            span._.rank = span._.similarity# * 1/(1 + np.exp(-span._.frequency))
            span._.span_id = ix
            span._.docs_idx = spans_docs_dict[span.text]
            span._.embedding = spans_embeddings[mmr_idx]
            top_spans.append(span)
        top_spans = sorted(top_spans, key=lambda span: span._.span_id)
        return top_spans
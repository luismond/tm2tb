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

        #docs = list(self.spacy_model.pipe(self.input_))
        self._set_span_rules(incl_pos, excl_pos)
        docs_spans = self._get_docs_spans(span_range)
        spans_dicts = self._get_spans_dicts(docs_spans, freq_min)
        docs_embeddings_avg = self._get_docs_embeddings_avg(spans_dicts.maps[1])
        spans_embeddings = trf_model.encode(list(spans_dicts.maps[2].keys()))
        sim_matrix = cosine_similarity(spans_embeddings, docs_embeddings_avg)
        best_spans_idx, candidate_spans_idx = self._get_mmr_idxs(spans_dicts, spans_embeddings, sim_matrix)
        top_spans = self._build_top_spans(sim_matrix, spans_dicts, spans_embeddings, best_spans_idx, candidate_spans_idx)
        if return_as_table is True:
            top_spans = self._return_as_table(top_spans)
        return top_spans

    @staticmethod
    def _set_span_rules(incl_pos=None, excl_pos=None):
        """
        Set custom attributes and properties on the document Spans.

        The attributes are used to filter the list of candidate spans.

        Tags are simple UPOS part-of-speech tags:

        ADJ: adjective
        ADP: adposition
        ADV: adverb
        AUX: auxiliary
        CCONJ: coordinating conjunction
        DET: determiner
        INTJ: interjection
        NOUN: noun
        NUM: numeral
        PART: particle
        PRON: pronoun
        PROPN: proper noun
        PUNCT: punctuation
        SCONJ: subordinating conjunction
        SYM: symbol
        VERB: verb
        X: other

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
            excl_pos = [tag for tag in pos_tags if tag not in incl_pos and not tag=='ADP']

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

    @staticmethod
    def _get_doc_spans(doc, span_range):
        """
        Select spans from a sentence using the given parameters and attributes.

        For example, it gets spans of length 1-3 whose tokens are nouns.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            Container representing one sentence
        span_range : tuple
            Represents the min/max span length range.

        Returns
        -------
        spans : list
            List of containers of type 'spacy.tokens.span.Span'

        """
        span_ranges = list((i, i+n) for i in range(len(doc))
                           for n in range(span_range[0], span_range[1]+1))
        spans = (doc[n:n_] for (n, n_) in span_ranges)

        return [sp for sp in spans
                if sp._.incl_pos_edges is True
                and sp._.excl_pos_any is True
                and sp._.alpha_edges is True
                and len(sp.text) > 1]

    def _get_docs_spans(self, span_range):
        """
        Iterate over `_get_doc_spans`. to get spans from each sentence.

        It unpacks all the retrieved spans to return a flattened list of spans.

        Parameters
        ----------
        docs : list
            List of spacy.tokens.doc.Doc
        span_range : tuple
            Represents the min/max span length range.

        Returns
        -------
        docs_spans : list
            List of 'spacy.tokens.span.Span'

        """
        docs_spans = [self._get_doc_spans(doc, span_range) for doc in self.docs]
        return set(itertools.chain(*docs_spans))

    def _get_spans_dicts(self, spans_list, freq_min=1):
        """
        Construct three dictionaries.

            `spans_freqs_dict`: All spans and their frequencies.
            `spans_docs_dict`: All spans and the documents in which they occur.
            `spans_texts_dict`: All spans and their string representation.

        Parameters
        ----------
        spans_list : list
            List of 'spacy.tokens.span.Span'
        docs : list
            List of spacy.tokens.doc.Doc
        freq_min : TYPE, optional
            Minimum frequency value. The default is 1.

        Raises
        ------
        ValueError
            Error raised when no spans are found above the minimum frequency.

        Returns
        -------
        spans_dicts : collections.ChainMap
            DESCRIPTION. Map representing the three dictionaries.

        """
        spans_freqs_dict = defaultdict(int)
        spans_docs_dict = defaultdict(set)
        spans_texts_dict = defaultdict(set)
        for span in spans_list:
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


    def _get_docs_embeddings_avg(self, spans_docs_dict):
        """
        Embed only the sentences in which candidate spans occur.

        Average the sentences embeddings to get a vector that represents the text.

        Parameters
        ----------
        text_docs : list
            List of spacy.tokens.doc.Doc
        spans_docs_dict : dict
            Mapping of all spans and the documents in which they occur.

        Returns
        -------
        docs_embeddings_avg: numpy.ndarray
            Array representing the the text embedding average.
        """
        top_docs_idx = set(itertools.chain(*list(spans_docs_dict.values())))
        top_docs_texts = list(self.docs[i].text for i in top_docs_idx)
        docs_embeddings = trf_model.encode([text for text in top_docs_texts if len(text) > 0])
        docs_embeddings_avg = sum(docs_embeddings)/len(docs_embeddings)
        return docs_embeddings_avg.reshape(1, -1)


    @staticmethod
    def _get_mmr_idxs(spans_dicts, spans_embeddings, spans_doc_sims):
        """
        Use Maximal Marginal Relevance to rank the spans.

        For now, diversity and top_n are hard coded.
        Top_n is the total number of candidates / 2.
        The default diversity value is .9
        All spans are kept, but the second-best spans are downweighted.

        Parameters
        ----------
        spans_dicts : collections.ChainMap
            See _get_spans_dict
        spans_embeddings : numpy.ndarray
            Array of embeddings of each span.
        spans_doc_sims : numpy.ndarray
            Array of similarity values of spans and the sentence/text.

        Returns
        -------
         best_spans_idx :
             List of indexes of best spans

         candidate_spans_idx :
             List of indexes of second-best spans

        """
        top_n = round(len(list(spans_dicts.maps[2].keys()))/2)
        diversity = .9
        # Construct a similarity matrix of spans
        spans_sims = cosine_similarity(spans_embeddings)
        # Choose best span to initialize the best spans index
        best_spans_idx = [np.argmax(spans_doc_sims)]
        # Initialize the candidates index
        candidates_idx = [i for i in range(len(spans_embeddings))
                          if i != best_spans_idx[0]]
        # Iteratively, select the best span, add it to best_spans_idx
        # and remove it from candidates_idx
        for _ in range(min(top_n - 1, len(spans_embeddings) - 1)):
            candidate_sims = spans_doc_sims[candidates_idx, :]
            rest_spans_sims = np.max(spans_sims[candidates_idx][:, best_spans_idx], axis=1)
            # Calculate Maximum Marginal Relevance
            mmr = (1-diversity) * candidate_sims - diversity * rest_spans_sims.reshape(-1, 1)
            # Get best candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]
            # Update best spans & candidates
            best_spans_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return best_spans_idx, candidates_idx

    @staticmethod
    def _build_top_spans(span_doc_similarities,
                         spans_dicts,
                         spans_embeddings,
                         best_spans_idx,
                         candidate_spans_idx):
        """
        Gather the generated data from the spans.
        Add the data to each span using custom span attributes.
        (See https://spacy.io/api/span)

        Parameters
        ----------
        span_doc_similarities : numpy.ndarray
            DESCRIPTION. Array representing similarity values
            between candidate spans and the document.

        spans_dicts : collections.ChainMap
            DESCRIPTION. See `_get_spans_dicts`

        spans_embeddings : numpy.ndarray
            DESCRIPTION. Array of embeddings for each span.

        best_spans_idx :
            DESCRIPTION. List of indexes of best spans

        candidate_spans_idx :
            DESCRIPTION. List of indexes of second-best spans

        Returns
        -------
        top_spans : list of spans of type 'spacy.tokens.span.Span'

        """
        spans_freqs_dict = spans_dicts.maps[0]
        spans_docs_dict = spans_dicts.maps[1]
        spans_texts_dict = spans_dicts.maps[2]

        # add best spans
        top_spans = []
        for idx in best_spans_idx:
            # Retrieve the span object
            span = list(spans_texts_dict.values())[idx]
            similarity = round(float(span_doc_similarities.reshape(1, -1)[0][idx]), 4)
            span._.similarity = similarity
            span._.frequency = spans_freqs_dict[span.text]
            span._.rank = span._.similarity * 1/(1 + np.exp(-span._.frequency))
            span._.span_id = idx
            span._.docs_idx = spans_docs_dict[span.text]
            span._.embedding = spans_embeddings[idx]
            top_spans.append(span)

        # add second-best spans, but downweight their rank
        for idx in candidate_spans_idx:
            span = list(spans_texts_dict.values())[idx]
            similarity = round(float(span_doc_similarities.reshape(1, -1)[0][idx]), 4)
            span._.similarity = similarity
            span._.frequency = spans_freqs_dict[span.text]
            span._.rank = (span._.similarity * 1/(1 + np.exp(-span._.frequency))) * .5
            span._.span_id = idx
            span._.docs_idx = spans_docs_dict[span.text]
            span._.embedding = spans_embeddings[idx]
            top_spans.append(span)

        return top_spans

    @staticmethod
    def _return_spans_as_tuples(spans):
        """Take a list spaCy spans, format it as named tuples."""
        Term = namedtuple('Term', ['term', 'similarity', 'frequency'])
        terms = []
        for span in spans:
            terms.append(Term(span.text, span._.similarity, span._.frequency))
        return terms

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

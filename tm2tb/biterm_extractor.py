"""
Extract bilingual terms from one or multiple sentences.

Classes:
    BitermExtractor

Functions:
    extract_terms(Union[tuple, list], **kwargs)
"""

from collections import defaultdict
from collections import namedtuple
from collections import ChainMap
from typing import Union
from itertools import groupby
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tm2tb import TermExtractor


class BitermExtractor:
    """

    Class representing a bilingual term/keyword extractor.

    Attributes
    ----------
    input_ : Union[tuple, list]
        DESCRIPTION. Tuple or list of tuples

    Methods
    -------
    extract_terms(self, similarity_min, **kwargs)

    """

    def __init__(self, input_: Union[tuple, list]):
        self.input_ = input_

    def extract_terms(self, similarity_min=.9, return_as_table=True, **kwargs):
        """

        Extract biterms from a pair of translated texts.

        If the input is a tuple, it calls `extract_terms_from_bisentence`.
        If the input is a list, it calls `extract_terms_from_bitext`.

        Parameters
        ----------
        similarity_min : int, optional
            Minimum similarity value between source and target terms.
            The default is .9.
        **kwargs : dict
            span_range : tuple, optional
                Length range of the terms. The default is (1, 3).

            freq_min : int, optional
                Minimum ocurrence frequency of the terms.
                The default is 1.

            incl_pos : list, optional
                See `TermExtractor._set_span_extensions`

            excl_pos : list, optional
                See `TermExtractor._set_span_extensions`

        Returns
        -------
        terms : list
            List of named tuples representing the extracted bilingual terms.
            For example:
                [BiTerm(src_term='world', src_tags=['NOUN'],
                       trg_term='mundo', trg_tags=['NOUN'],
                       similarity=0.9823, frequency=1)]
        """
        if isinstance(self.input_, tuple):
            terms = self.extract_terms_from_bisentence(similarity_min, **kwargs)
        if isinstance(self.input_, list):
            terms = self.extract_terms_from_bitext(similarity_min, **kwargs)
        if return_as_table is True:
            terms = self._return_as_table(terms)
        return terms

    def extract_terms_from_bisentence(self, similarity_min, **kwargs):
        """

        Extract bilingual terms from a source sentence and a target sentence.

        Call the class TermExtractor to extract terms from each sentence.
        Compare the embeddings of both terms lists and align them.

        Parameters
        ----------
        similarity_min : int
            Minimum similarity value between source and target terms.
        **kwargs : dict
            See `extract_terms`

        Returns
        -------
        biterms : list
            List of named tuples representing the extracted bilingual terms.

        """
        bisentence = self.input_

        src_sentence = bisentence[0]
        src_extractor = TermExtractor(src_sentence)
        src_terms = src_extractor.extract_terms(return_as_table=False, **kwargs)
        src_terms = sorted(src_terms, key=lambda span: span._.span_id)

        trg_sentence = bisentence[1]
        trg_extractor = TermExtractor(trg_sentence)
        trg_terms = trg_extractor.extract_terms(return_as_table=False, **kwargs)
        trg_terms = sorted(trg_terms, key=lambda span: span._.span_id)

        similarity_matrix = self._get_similarity_matrix(src_terms, trg_terms)

        biterms_freqs_dict = defaultdict(int)
        biterms_sims_dict = defaultdict(set)
        biterms_spans_dict = defaultdict(set)

        for src_term, trg_term in list(product(src_terms, trg_terms)):
            similarity = similarity_matrix[src_term._.span_id,
                                           trg_term._.span_id]
            if similarity > similarity_min:
                biterms_freqs_dict[(src_term.text, trg_term.text)] += 1
                biterms_sims_dict[(src_term.text, trg_term.text)] = round(similarity, 4)
                biterms_spans_dict[(src_term.text, trg_term.text)] = (src_term, trg_term)

        biterms_dicts = ChainMap(biterms_freqs_dict, biterms_sims_dict, biterms_spans_dict)
        biterms = self._build_biterms(biterms_dicts)
        biterms = self._prune_biterms(biterms, 'src')
        biterms = self._prune_biterms(biterms, 'trg')
        biterms = sorted(biterms, key=lambda biterm: biterm.similarity, reverse=True)
        return biterms

    def extract_terms_from_bitext(self, similarity_min, **kwargs):
        """

        Extract biterms from a bitext (multiple bilingual sentences).

        It does not iterate over each bisentence, since it would be slow.
        It calls the TermExtractor class to get *all* the source/target terms.
        It produces a similarity matrix of all source and target terms.
        It looks up the biterm co-ocurrences
        (biterms that occur in the same bisentence)
        It retrieves the most similar biterms from the co-ocurrrent biterms.
        It keeps the best target term for each source term and viceversa.

        Parameters
        ----------
        similarity_min : int
            Minimum similarity value between the source and the target terms.
        **kwargs : dict
            See `extract_terms`

        Returns
        -------
        biterms : list
            List of named tuples representing the extracted bilingual terms.

        """
        bitext = self.input_

        def validate_if_bitext_is_bilingual(bitext):
            """
            If source and target rows are equal, it is pointless to extract biterms.
            Return bitext if source and target rows are different.
            """
            n = 0
            for row in bitext:
                if row[0]==row[1]:
                    n += 1
            if len(bitext) == n:
                raise ValueError('Source rows are identical to target rows!')
            #else:
            return bitext

        bitext = validate_if_bitext_is_bilingual(bitext)
        src_text, trg_text = zip(*bitext)

        src_extractor = TermExtractor(list(src_text))
        src_terms = src_extractor.extract_terms(return_as_table=False, **kwargs)

        trg_extractor = TermExtractor(list(trg_text))
        trg_terms = trg_extractor.extract_terms(return_as_table=False, **kwargs)

        similarity_matrix = self._get_similarity_matrix(src_terms, trg_terms)
        bitext_spans_dict = self._get_bitext_spans_dict(src_terms, trg_terms)

        biterms_freqs_dict = defaultdict(int)
        biterms_sims_dict = defaultdict(set)
        biterms_spans_dict = defaultdict(set)

        for src_trg_spans in bitext_spans_dict.values():
            src_spans = src_trg_spans[0]
            trg_spans = src_trg_spans[1]
            for src_span, trg_span in list(product(src_spans, trg_spans)):
                similarity = similarity_matrix[src_span._.span_id, trg_span._.span_id]
                if similarity > similarity_min:
                    biterms_freqs_dict[(src_span.text, trg_span.text)] += 1
                    biterms_sims_dict[(src_span.text, trg_span.text)] = round(similarity, 4)
                    biterms_spans_dict[(src_span.text, trg_span.text)] = (src_span, trg_span)

        biterms_dicts = ChainMap(biterms_freqs_dict, biterms_sims_dict, biterms_spans_dict)
        biterms = self._build_biterms(biterms_dicts)
        biterms = self._prune_biterms(biterms, 'src')
        biterms = self._prune_biterms(biterms, 'trg')
        biterms = sorted(biterms, key=lambda biterm: biterm.biterm_rank, reverse=True)
        return biterms

    @staticmethod
    def _get_similarity_matrix(src_spans, trg_spans):
        """
        Generate a similarity matrix of source and target term candidates.

        Parameters
        ----------
        src_spans : list
            List of spans of type 'spacy.tokens.span.Span' from the source side
        trg_spans : list
            List of spans of type 'spacy.tokens.span.Span' from the target side

        Returns
        -------
        similarity_matrix : numpy.ndarray
            Similarity matrix produced by calculating the cosine similarity
                         of the source and the target spans embeddings.

        """
        src_embeddings = [span._.embedding for span in src_spans]
        trg_embeddings = [span._.embedding for span in trg_spans]
        similarity_matrix = cosine_similarity(src_embeddings, trg_embeddings)
        return similarity_matrix

    @staticmethod
    def _get_bitext_spans_dict(src_spans, trg_spans):
        """
        Construct a mapping of all the bisentences indices and their biterms.

        (When extracting biterms from multiple bisentences)

        """
        src_doc_spans_dict = defaultdict(set)
        for span in src_spans:
            for doc_id in span._.docs_idx:
                src_doc_spans_dict[doc_id].add(span)

        trg_doc_spans_dict = defaultdict(set)
        for span in trg_spans:
            for doc_id in span._.docs_idx:
                trg_doc_spans_dict[doc_id].add(span)

        bitext_spans_dict = defaultdict(list)
        for doc_id, span in src_doc_spans_dict.items():
            bitext_spans_dict[doc_id].append(span)
        for doc_id, span in trg_doc_spans_dict.items():
            bitext_spans_dict[doc_id].append(span)

        # Keep only bitext rows with span candidates on both sides.
        bitext_spans_dict = {k: v for (k, v) in bitext_spans_dict.items() if len(v) > 1}
        return bitext_spans_dict

    @staticmethod
    def _prune_biterms(biterms, side):
        """
        Keep the most similar target term for each source term (and viceversa).

        Parameters
        ----------
        biterms : list
            List of named tuples.
        side : str
            String representing which "side" to prune: "src" or "trg".

        Returns
        -------
        biterms_ : list
            Pruned list of named tuples in the same format.
        """
        if side == 'src':
            keyfunc = lambda k: k.src_term
            biterms = sorted(biterms, key=keyfunc)
        if side == 'trg':
            keyfunc = lambda k: k.trg_term
            biterms = sorted(biterms, key=keyfunc)

        biterms_ = []
        for _, group in groupby(biterms, keyfunc):
            # Sort biterms group by similarity
            group = sorted(group, key=lambda group: group.similarity, reverse=True)
            # Take first biterm
            best_biterm = list(group)[0]
            biterms_.append(best_biterm)
        return biterms_

    @staticmethod
    def _build_biterms(biterms_dicts):
        """
        Gather the metadata from the biterms and build the final biterms list.

        Parameters
        ----------
        biterms_dicts : collections.ChainMap
        Map containing three dictionaries:
            biterms_freqs_dict: Biterms and their frequencies
            biterms_sims_dict: Bterms and their similarities
            biterms_spans_dict: Biterms and their string representations.

        Returns
        -------
        biterms : list
            List of named tuples representing the extracted bilingual terms.
            For example:
                [BiTerm(src_term='world', src_tags=['NOUN'],
                       trg_term='mundo', trg_tags=['NOUN'],
                       similarity=0.9823, frequency=1)]

        """
        biterms_freqs_dict = biterms_dicts.maps[0]
        biterms_sims_dict = biterms_dicts.maps[1]
        biterms_spans_dict = biterms_dicts.maps[2]

        BiTerm = namedtuple('BiTerm', ['src_term', 'src_tags', 'src_rank',
                                       'trg_term', 'trg_tags', 'trg_rank',
                                       'similarity', 'frequency', 'biterm_rank'])

        def get_biterm_rank(frequency, similarity, src_rank, trg_rank):
            freq_sigm = 1/(1 + np.exp(-frequency))
            # downweight non-translatables (terms which are equal)
            if similarity == 1:
                similarity = similarity * .7
            biterm_rank = ((src_rank + trg_rank)/2) * similarity * freq_sigm
            biterm_rank = round((1/(1 + np.exp(-biterm_rank))), 4)
            return biterm_rank

        biterms = []
        for biterm_, similarity in biterms_sims_dict.items():
            src_term = biterm_[0]
            trg_term = biterm_[1]
            spans = biterms_spans_dict[biterm_]
            src_span = spans[0]
            trg_span = spans[1]
            src_tags = [t.pos_ for t in src_span]
            trg_tags = [t.pos_ for t in trg_span]
            src_rank = src_span._.rank
            trg_rank = trg_span._.rank
            frequency = biterms_freqs_dict[biterm_]
            biterm_rank = get_biterm_rank(frequency, similarity, src_rank, trg_rank)

            biterm = BiTerm(src_term, src_tags, src_rank,
                            trg_term, trg_tags, trg_rank,
                            similarity, frequency, biterm_rank)

            biterms.append(biterm)
        if len(biterms)==0:
            raise ValueError('No biterms found.')
        return biterms

    @staticmethod
    def _return_as_table(biterms):
        """ Return biterms as pandas dataframe."""
        biterms = pd.DataFrame(biterms)
        biterms = biterms.sort_values(by='biterm_rank', ascending=False)
        biterms.reset_index(drop=True, inplace=True)
        biterms = biterms.round(4)
        return biterms

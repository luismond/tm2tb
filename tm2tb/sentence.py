"""
Sentence class
"""
from collections import defaultdict
from collections import namedtuple
from langdetect import detect
from spacy.tokens import Span
from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from tm2tb import get_spacy_model
from tm2tb import trf_model

class Sentence:
    supported_languages = ['en', 'es', 'fr', 'de']
    def __init__(self, sentence):
        self.sentence = sentence
        self.lang = self._get_lang()
        Span.set_extension("similarity", default=None, force=True)
        Span.set_extension("span_id", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        self.doc = self._make_doc()

    def _get_lang(self):
        lang = detect(self.sentence)
        if lang not in self.supported_languages:
            raise ValueError('Language not supported!')
        return lang

    def _make_doc(self):
        """
        Construct a Doc object from the sentence.
        Get doc embedding and add it to doc._.embedding.
        """
        Doc.set_extension("embedding", default=None, force=True)
        spacy_model = get_spacy_model(self.lang)
        doc = spacy_model(self.sentence)
        doc._.embedding = trf_model.encode([doc.text])
        return doc

    @staticmethod
    def _set_span_extensions(incl_pos, excl_pos):
        # Get incl_pos and/or excl_pos or use default tags
        if incl_pos is None:
            incl_pos = ['NOUN', 'PROPN', 'ADJ']
        if excl_pos is None:
            excl_pos = ['VERB', 'CCONJ', 'ADP', 'PUNCT']
        # Define span pos rules
        incl_pos_getter = lambda span: span[0].pos_ in incl_pos and span[-1].pos_ in incl_pos
        excl_pos_getter = lambda span: not any(t.pos_ in excl_pos for t in span)
        alpha_edges_getter = lambda span: span[0].text.isalpha() and span[-1].text.isalpha()
        # Register span extensions
        Span.set_extension("edges_incl_pos", getter=incl_pos_getter, force=True)
        Span.set_extension("any_excl_pos", getter=excl_pos_getter, force=True)
        Span.set_extension("edges_are_alpha", getter=alpha_edges_getter, force=True)

    def _filter_spans(self, span_min_len=1, span_max_len=2, incl_pos=None, excl_pos=None):
        # Get span ranges
        span_ranges = list((i,i+n) for i in range(len(self.doc))
          for n in range(span_min_len, span_max_len+1))
        # Get all spans from doc within ranges
        spans = (self.doc[n:n_] for (n, n_) in span_ranges)
        # Filter resulting spans with span extension methods
        self._set_span_extensions(incl_pos, excl_pos)
        spans = [sp for sp in spans if
                 sp._.edges_incl_pos is True and
                 sp._.any_excl_pos is True and
                 sp._.edges_are_alpha is True]
        if len(spans)==0:
            raise ValueError('No spans left after filtering!')
        return spans

    def get_top_terms(self, return_as_tuples=True, **kwargs):
        """
        Get span candidates, sort them by similarity to the doc itself.
        """
        spans = self._filter_spans(**kwargs)
        # Make text->span mapping
        text_span_dict = defaultdict(set)
        for span in spans:
            text_span_dict[span.text] = span
        # Get spans embeddings
        span_embeddings = trf_model.encode(list(text_span_dict.keys()))
        # Get span/doc similarities
        span_doc_sims = cosine_similarity(self.doc._.embedding, span_embeddings)
        # Register span similarity extensions
        Span.set_extension("similarity", default=None, force=True)
        Span.set_extension("span_id", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        # Add similarity value and embedding to each span
        top_spans = []
        for idx, span in enumerate(text_span_dict.values()):
            span._.span_id = idx
            span._.similarity = round(float(span_doc_sims.reshape(1, -1)[0][idx]), 4)
            span._.embedding = span_embeddings[idx]
            top_spans.append(span)
        if return_as_tuples is True:
            top_spans = self._return_spans_as_tuples(top_spans)
        return top_spans

    @staticmethod
    def _return_spans_as_tuples(spans):
        NamedSpan = namedtuple('Term', ['term', 'similarity'])
        named_spans = []
        for span in spans:
            named_spans.append(NamedSpan(span.text, span._.similarity))
        return sorted(named_spans, key=lambda span: span.similarity, reverse=True)

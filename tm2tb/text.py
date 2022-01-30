"""
Text class
"""
from random import randint
from typing import Tuple
from collections import Counter as cnt
from collections import defaultdict
from collections import namedtuple
import itertools
import numpy as np
from langdetect import detect, LangDetectException
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Span
from tm2tb import trf_model
from tm2tb.spacy_models import get_spacy_model
from tm2tb.preprocess import preprocess

class Text:
    """
    Class that represents text data and term extraction methods
    Class that represents a set of spaCy docs
    """
    supported_languages = ['en', 'es', 'de', 'fr']
    def __init__(self, text):
        self.text = text
        self.docs = self.get_docs()
        self.spans_freqs_dict = defaultdict(int)
        self.spans_docs_dict = defaultdict(set)
        self.spans_strings_dict = defaultdict(set)

    @staticmethod
    def _preprocess_text(text):
        text_clean = []
        for sentence in text:
            try:
                text_clean.append(preprocess(sentence))
            except ValueError:
                pass
        if len(text_clean)==0:
            raise ValueError('No clean sentences!')
        return text_clean

    @staticmethod
    def _detect_text_lang(text, supported_languages, sample_len=20):
        sentences = [s for s in text if len(s)>20]
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
        if lang not in supported_languages:
            raise ValueError('Lang not supported!')
        return lang

    def get_docs(self):
        """
        Pipes all text strings to the spaCy model
        """
        text_clean = self._preprocess_text(self.text)
        lang = self._detect_text_lang(text_clean, self.supported_languages)
        spacy_model = get_spacy_model(lang)
        docs = list((spacy_model.pipe(text_clean)))
        return docs

    @staticmethod
    def get_doc_spans(doc, span_range):
        """
        Get spans from one doc
        """
        span_min_len = span_range[0]
        span_max_len = span_range[1]
        span_ranges = list((i,i+n) for i in range(len(doc))
                  for n in range(span_min_len, span_max_len+1))
        spans = (doc[n:n_] for (n, n_) in span_ranges)
        spans = (sp for sp in spans if sp._.edges_include_pos is True
                 and sp._.any_exclude_pos is True
                 and sp._.edges_are_alpha is True)
        return spans

    def get_docs_spans(self, span_range: Tuple[int, int] = (1,2)):
        """
        Gets spans from all docs in Text filtered by range and pos tags
        """
        include_pos = ['NOUN', 'PROPN', 'ADJ']
        exclude_pos = ['VERB', 'CCONJ', 'ADP', 'PUNCT']
        include_pos_getter = lambda span: span[0].pos_ in include_pos \
            and span[-1].pos_ in include_pos
        exclude_pos_getter = lambda span: not any(t.pos_ in exclude_pos for t in span)
        alpha_edges_getter = lambda span: span[0].text.isalpha() and span[-1].text.isalpha()
        Span.set_extension("edges_include_pos", getter=include_pos_getter, force=True)
        Span.set_extension("any_exclude_pos", getter=exclude_pos_getter, force=True)
        Span.set_extension("edges_are_alpha", getter=alpha_edges_getter, force=True)
        doc_spans = (self.get_doc_spans(doc, span_range) for doc in self.docs)
        docs_spans = set(itertools.chain(*doc_spans))
        return docs_spans

    def update_dicts(self, span_range, freq_min=1):
        """
        Update class dicts
        """
        spans = self.get_docs_spans(span_range)
        for span in spans:
            # get span.text frequencies
            self.spans_freqs_dict[span.text] += 1
            # get span.text-span mapping
            self.spans_strings_dict[span.text] = span
            # add all document indices to span-doc dictionary
            # Get all doc indices of documents where each span occurs
            self.spans_docs_dict[span.text].add(self.docs.index(span.doc))

        for span_text, freq in self.spans_freqs_dict.items():
            if freq < freq_min:
                self.spans_docs_dict.pop(span_text)
                self.spans_strings_dict.pop(span_text)

    def get_docs_embeddings_avg(self):
        """
        Gets docs embeddings average
        """
        top_docs_idx = set(itertools.chain(*list(self.spans_docs_dict.values())))
        top_docs_texts = list(self.docs[i].text for i in top_docs_idx)
        docs_embeddings = trf_model.encode(top_docs_texts)
        docs_embeddings_avg = sum(docs_embeddings)/len(docs_embeddings)
        return docs_embeddings_avg.reshape(1, -1)

    def get_spans_embeddings(self):
        """
        Gets embeddings of span candidates
        """
        top_spans_strings = list(self.spans_strings_dict.keys())
        spans_embeddings = trf_model.encode(top_spans_strings)
        return spans_embeddings

    @staticmethod
    def get_rank_sigmoid(freq, sim):
        """
        Squashes the product of the span's frequency and similarity with a sigmoid function
        """
        rank = 1/(1 + np.exp(-freq*sim))
        rank = round(rank, 4)
        return rank

    def get_top_spans(self, span_range=(1,3), freq_min=1, return_embeddings=False):
        """
        Use document embedding and span candidates embeddings to obtain their similarity
        Add frequency, rank, doc_idx and similarity metadata to spans
        """
        self.update_dicts(span_range, freq_min)

        Span.set_extension("sim", default=None, force=True)
        Span.set_extension("rank", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        Span.set_extension("freq", default=None, force=True)
        Span.set_extension("docs_idx", default=None, force=True)

        # Get doc/span similarities
        docs_embeddings_avg = self.get_docs_embeddings_avg()
        spans_embeddings = self.get_spans_embeddings()
        span_doc_sims = cosine_similarity(spans_embeddings, docs_embeddings_avg)
        span_doc_sims_idx = list(range(len(span_doc_sims)))

        # Build top spans with their frequency, similarity, rank, etc.
        top_spans = []
        for idx in span_doc_sims_idx:
            span = list(self.spans_strings_dict.values())[idx]
            sim = round(float(span_doc_sims.reshape(1, -1)[0][idx]), 4)
            if sim > 0:
                span._.sim = sim
                span._.freq = self.spans_freqs_dict[span.text]
                span._.rank = self.get_rank_sigmoid(span._.freq, span._.sim)
                span._.docs_idx = self.spans_docs_dict[span.text]
                span_tags = [t.pos_ for t in span]
                if return_embeddings is True:
                    TopSpan = namedtuple('Term', ['term',
                                                  'doc_similarity',
                                                  'frequency',
                                                  'rank',
                                                  'docs_idx',
                                                  'tags',
                                                  'emb'])

                    span._.embedding = spans_embeddings[idx]
                    top_span = TopSpan(span.text,
                                       span._.sim,
                                       span._.freq,
                                       span._.rank,
                                       span._.docs_idx,
                                       span_tags,
                                       span._.embedding)
                if return_embeddings is False:
                    TopSpan = namedtuple('Term', ['term',
                                                  'doc_similarity',
                                                  'frequency',
                                                  'rank',
                                                  'tags'])
                    top_span = TopSpan(span.text,
                                       span._.sim,
                                       span._.freq,
                                       span._.rank,
                                       span_tags)
                top_spans.append(top_span)
        top_spans = sorted(top_spans, key=lambda top_span: top_span.rank, reverse=True)
        return top_spans

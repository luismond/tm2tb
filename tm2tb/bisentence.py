"""
BiSentence class.
"""
from collections import namedtuple
from itertools import groupby
from tm2tb import Sentence
from sklearn.metrics.pairwise import cosine_similarity

class BiSentence:
    def __init__(self, bisentence):
        self.bisentence = bisentence
        self.src_sentence = self.bisentence[0]
        self.trg_sentence = self.bisentence[1]
        self.src_sn = Sentence(self.src_sentence)
        self.trg_sn = Sentence(self.trg_sentence)

    @staticmethod
    def _get_similarity_matrix(src_top_spans, trg_top_spans):
        src_embeddings = [sp._.embedding for sp in src_top_spans]
        trg_embeddings = [sp._.embedding for sp in trg_top_spans]
        sim_matrix = (cosine_similarity(src_embeddings, trg_embeddings))
        return sim_matrix

    def get_top_biterms(self, **kwargs):
        """
        Get top spans from both sentences.
        Calculate their similarity and align them.
        """
        src_top_spans = self.src_sn.get_top_terms(return_as_tuples=False, **kwargs)
        trg_top_spans = self.trg_sn.get_top_terms(return_as_tuples=False, **kwargs)
        similarity_matrix = self._get_similarity_matrix(src_top_spans, trg_top_spans)
        top_biterms = []
        BiTerm = namedtuple('BiTerm', ['src_term', 'trg_term', 'similarity'])
        for src_span in src_top_spans:
            for trg_span in trg_top_spans:
                sim = round(similarity_matrix[src_span._.span_id,
                                             trg_span._.span_id], 4)
                if sim > .9:
                    biterm = BiTerm(src_span.text, trg_span.text, sim)
                    top_biterms.append(biterm)
        if len(top_biterms)==0:
            raise ValueError('No biterms found in bisentence!')
        return self._deduplicate_biterms(top_biterms)

    @staticmethod
    def _deduplicate_biterms(biterms):
        groups_src = []
        keyfunc = lambda k: k.src_term
        biterms = sorted(biterms, key=keyfunc)
        for _, g in groupby(biterms, keyfunc):
            g = sorted(g, key=lambda g: g.similarity, reverse=True)
            top = list(g)[0]
            groups_src.append(top)

        groups_trg = []
        keyfunc = lambda k: k.trg_term
        groups_src = sorted(groups_src, key=keyfunc)
        for _, g in groupby(groups_src, keyfunc):
            g = sorted(g, key=lambda g: g.similarity, reverse=True)
            top = list(g)[0]
            groups_trg.append(top)

        return sorted(groups_trg, key=lambda x:x.similarity, reverse=True)

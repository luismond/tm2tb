from collections import defaultdict
from collections import namedtuple
from itertools import groupby
from tm2tb import Text
from sklearn.metrics.pairwise import cosine_similarity

class BiText:
    def __init__(self, bitext):
        self.src_text = Text(bitext['src'].tolist())
        self.trg_text = Text(bitext['trg'].tolist())
        self.src_top_spans = self.src_text.get_top_terms(return_as_tuples=False)
        self.trg_top_spans = self.trg_text.get_top_terms(return_as_tuples=False)
        self.bitext_top_spans_dict = self._get_bitext_top_spans_dict()

    def _get_biterms_similarity_matrix(self):
        src_embeddings = [sp._.embedding for sp in self.src_top_spans]
        trg_embeddings = [sp._.embedding for sp in self.trg_top_spans]
        sim_matrix = (cosine_similarity(src_embeddings, trg_embeddings))
        return sim_matrix

    def _get_bitext_top_spans_dict(self):
        # Get inverse doc-> spans src dict
        src_doc_spans_dict = defaultdict(set)
        for span in self.src_top_spans:
            for i in span._.docs_idx:
                src_doc_spans_dict[i].add(span)

        trg_doc_spans_dict = defaultdict(set)
        for span in self.trg_top_spans:
            for i in span._.docs_idx:
                trg_doc_spans_dict[i].add(span)

        # Get dict with bitext rows as keys and src/trg top spans as values
        bitext_top_spans_dict = defaultdict(list)
        for k, v in src_doc_spans_dict.items():
            bitext_top_spans_dict[k].append(v)
        for k, v in trg_doc_spans_dict.items():
            bitext_top_spans_dict[k].append(v)
        return bitext_top_spans_dict

    def get_top_biterms(self):
        """
        Align biterms from each besentence in bitext.
        Return a list of named tuples in the format:
        BiTerm(src_term='world', trg_term='mundo', sim=0.99, freq=4)
        """
        biterm_sim_matrix = self._get_biterms_similarity_matrix()
        top_biterms_freqs_dict = defaultdict(int)
        top_biterms_sims_dict = defaultdict(set)
        for tup in self.bitext_top_spans_dict.values():
            if len(tup)>1:
                for trg_span in tup[1]:
                    for src_span in tup[0]:
                        sim = biterm_sim_matrix[src_span._.span_id,
                                                trg_span._.span_id]
                        top_biterms_freqs_dict[(src_span.text, trg_span.text)] +=1
                        top_biterms_sims_dict[(src_span.text, trg_span.text)] = round(sim, 4)

        BiTerm = namedtuple('BiTerm', ['src_term', 'trg_term', 'sim', 'freq'])
        biterms = []
        for k, v in top_biterms_sims_dict.items():
            if v > .9:
                biterm = BiTerm(k[0], k[1], v, top_biterms_freqs_dict[k])
                biterms.append(biterm)
        biterms = self._get_top_matches(biterms)
        return biterms

    @staticmethod
    def _get_top_matches(biterms):
        groups_src = []
        keyfunc = lambda k: k.src_term
        biterms = sorted(biterms, key=keyfunc)
        for _, g in groupby(biterms, keyfunc):
            g = sorted(g, key=lambda g: g.sim, reverse=True)
            top = list(g)[0]
            groups_src.append(top)
        groups_trg = []
        keyfunc = lambda k: k.trg_term
        groups_src = sorted(groups_src, key=keyfunc)
        for _, g in groupby(groups_src, keyfunc):
            g = sorted(g, key=lambda g: g.sim, reverse=True)
            top = list(g)[0]
            groups_trg.append(top)
        return groups_trg

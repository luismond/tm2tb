"""
BiText class
"""
from tm2tb import Text
from tm2tb.align_ngrams import get_top_ngrams

class BiText:
    def __init__(self, bitext):
        self.src_text = Text(bitext['src'].tolist())
        self.trg_text = Text(bitext['trg'].tolist())

    def get_ngrams_dfs(self, **kwargs):
        src_ngrams_df = self.src_text.get_top_ngrams(return_embs=True,
                                                           **kwargs)
        trg_ngrams_df = self.trg_text.get_top_ngrams(return_embs=True,
                                                           **kwargs)
        return src_ngrams_df, trg_ngrams_df
    
    def get_top_ngrams(self, **kwargs):
        src_ngrams_df, trg_ngrams_df = self.get_ngrams_dfs(**kwargs)
        top_ngrams = get_top_ngrams(src_ngrams_df,
                                    trg_ngrams_df,
                                    min_similarity=.8,
                                    **kwargs)
        return top_ngrams

"""
BiSentence class.
"""
from tm2tb import Sentence
from tm2tb.align_ngrams import get_top_ngrams

class BiSentence:
    def __init__(self, sentence_tuple):
        self.src_sentence = Sentence(sentence_tuple[0])
        self.trg_sentence = Sentence(sentence_tuple[1])

    def get_ngrams_dfs(self, **kwargs):
        src_ngrams_df = self.src_sentence.get_top_ngrams(return_embs=True,
                                                         **kwargs)
        trg_ngrams_df = self.trg_sentence.get_top_ngrams(return_embs=True,
                                                         **kwargs)
        return src_ngrams_df, trg_ngrams_df
    
    def get_top_ngrams(self, **kwargs):
        src_ngrams_df, trg_ngrams_df = self.get_ngrams_dfs(**kwargs)
        top_ngrams = get_top_ngrams(src_ngrams_df,
                                    trg_ngrams_df,
                                    min_similarity=.8,
                                    **kwargs)
        return top_ngrams

#%% -*- coding: utf-8 -*-
"""
Main application function. It takes a bilingual file and generates a bilingual terminology list
"""
from collections import Counter as cnt
import warnings
import pandas as pd
from lib.preprocess import preprocess, strip_punct
from lib.read_bilingual_file import BilingualReader
from lib.my_word_vectorizer import my_word_vectorizer
from lib.tm_iter_zip import tm_iter_zip
from lib.tb_functions import tb_prune_fqs, get_mt_match
from lib.tmtb_to_html import tmtb_to_html
from lib.tb_to_azure import tb_to_azure
warnings.filterwarnings("ignore")

def tm2tb_main(filename):
    """
    Parameters
    ----------
    filename : string
        name of the file.
    Returns
    -------
    TYPE: html string
        Html string that represents the bitext and the term list as tables for preview in the app
    """
    upload_path = 'uploads'
    try:
        bitext = BilingualReader(upload_path, filename).get_bitext()
    except ValueError as value_error:
        return str('<p class="text-center error">{}</p>'.format(value_error))
    bitext, srcdet, trgdet = preprocess(bitext)
    # MY gensim WORD VECTORIZER. Get gensim vector objects
    bitext, src_dict, trg_dict, stmd, ttmd = my_word_vectorizer(bitext)
    # TM ITERZIP. Generate all possible source-target term combinations from segments
    bitext = tm_iter_zip(bitext)
    # GET FREQ DICT OF PAIR CANDIDATES
    tbc = dict(cnt([pair for segment in bitext['iter_zip'].tolist() for pair in segment]))
    # GET TB DF
    term_base = pd.DataFrame(zip(tbc.keys(), tbc.values()))
    term_base.columns = ['pair', 'pair_fq']
    # PAIR CANDIDATES ABOVE N
    term_base = term_base[term_base['pair_fq']>1]
    # CREATE SRC AND TRG COLUMNS
    term_base['src'] = term_base['pair'].apply(lambda tup: tup[0])
    term_base['trg'] = term_base['pair'].apply(lambda tup: tup[1])
    # DROP SHORT SEGMENTS
    term_base = term_base[term_base['src'].str.len()>3]
    term_base = term_base[term_base['trg'].str.len()>3]
    # GET TERM FREQUENCY
    term_base['src_fq'] = term_base['src'].apply(lambda i: src_dict.cfs[src_dict.token2id[i]])
    term_base['trg_fq'] = term_base['trg'].apply(lambda i: trg_dict.cfs[trg_dict.token2id[i]])
    # GET TERM TFIDF
    term_base['src_tfidf'] = term_base['src'].apply(lambda i: stmd[src_dict.token2id[i]])
    term_base['trg_tfidf'] = term_base['trg'].apply(lambda i: ttmd[trg_dict.token2id[i]])
    term_base = tb_prune_fqs(term_base)
    # GET NON TRANSLATABLES TB
    tb_nt = term_base[term_base['src']==term_base['trg']]
    # ADD 'ORIGIN' COLUMN
    tb_nt['origin'] = 'nt'
    # DROP NON TRANS FROM TB
    term_base = term_base[term_base['src'].isin(tb_nt['src']) == False]
    term_base = term_base[term_base['trg'].isin(tb_nt['trg']) == False]
    # SEND TB TO AZURE
    term_base = tb_to_azure(term_base, srcdet, trgdet)
    # CLEAN MT RESULTS
    term_base['mt_cands'] = term_base['mt_cands'].apply(lambda l: [strip_punct(w) for w in l])
    # FIND IF ANY MT RESULT MATCHES TRG CANDIDATE
    term_base['z'] = list(zip(term_base['mt_cands'], term_base['trg']))
    # GET MT MATCHES
    term_base['mt_match'] = term_base['z'].apply(get_mt_match)
    # DROP NAN AND HELPER COLUMNS
    term_base = term_base.dropna()
    term_base = term_base.drop(columns=['z','mt_match'])
    # ADD 'ORIGIN' COLUMN
    term_base['origin'] = 'mt'
    # CONCAT MT TB AND NT TB
    term_base = pd.concat([tb_nt, term_base])
    term_base = term_base.sort_values(by='pair_fq',ascending=False)
    term_base = pd.DataFrame(zip(term_base['src'],term_base['trg']))
    term_base.columns=['src','trg']
    term_base = term_base[term_base['src'].str.len()>3]
    term_base = term_base[term_base['trg'].str.len()>3]
    term_base.to_csv('{}/{}_tb.csv'.format(upload_path, filename), encoding='utf8', index=False)
    #CONVERT TO HTML FOR PREVIEW
    tmtb_html = tmtb_to_html(upload_path, filename, bitext, term_base)
    return tmtb_html

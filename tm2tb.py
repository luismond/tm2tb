#%% -*- coding: utf-8 -*-
from time import time
from langdetect import detect
from collections import Counter as cnt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
    
# PREPROCESSING
from lib.preprocess import mostly_alpha
from lib.preprocess import html_check
from lib.preprocess import pad_camels
from lib.preprocess import clean_special_symbols
from lib.preprocess import pad_punct
from lib.preprocess import drop_double_spaces
from lib.preprocess import strip_punct

from lib.fn_to_df import fn_to_df
from lib.get_stopwords import get_stopwords
from lib.get_grams import get_grams
from lib.my_word_vectorizer import my_word_vectorizer
from lib.tm_iter_zip import tm_iter_zip
from lib.tb_functions import tb_prune_fqs, get_mt_match

#HTML
from lib.tmtb_to_html import tmtb_to_html

#MT and lookup
# from lib.marian_mt import marian_mt
from lib.tb_to_azure import tb_to_azure

#TODO: short segments, prune subterms, evaluate capitalization

def tm2tb_main(filename):
    #try:
    start = time()
    # FILENAME TO DATAFRAME
    upload_path = 'uploads'
    tm = fn_to_df(upload_path, filename)
    print('{}: {}'.format('tm len',len(tm)))
    print('{} {}'.format('fn to df',time()-start))
    # PREPROCESS. Handle html, punctuation and other dirty characters.
    #tm = preprocess(tm)
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    # KEEP MOSTLY ALPHA
    tm['src'] = tm['src'].apply(mostly_alpha)
    tm['trg'] = tm['trg'].apply(mostly_alpha)
    tm = tm.dropna()
    tm = tm.astype(str)
    #tm = tm.drop_duplicates()
    # PRESERVE ORIGINAL STRINGS
    tm['srcu'] = tm['src']
    tm['trgu'] = tm['trg']
    # CLEAN HTML
    tm['src'] = tm['src'].apply(lambda s: html_check(s))
    tm['trg'] = tm['trg'].apply(lambda s: html_check(s))
    # FIX CAMELCASE like this: HelloWorld -> Hello World
    tm['src'] = tm['src'].apply(pad_camels)
    tm['trg'] = tm['trg'].apply(pad_camels)
    # REPLACE newlines and tabs \n, \t, \r with space
    tm['src'] = tm['src'].apply(clean_special_symbols)
    tm['trg'] = tm['trg'].apply(clean_special_symbols)
    # CORRECT ELLIPSIS
    # tm['src'] = tm['src'].apply(correct_ellipsis)
    # tm['trg'] = tm['trg'].apply(correct_ellipsis)
    # PAD PUNCT
    tm['src'] = tm['src'].apply(pad_punct)
    tm['trg'] = tm['trg'].apply(pad_punct)
    # CLEAN DOUBLE SPACES
    tm['src'] = tm['src'].apply(drop_double_spaces)
    tm['trg'] = tm['trg'].apply(drop_double_spaces)
    # DROP EMPTY
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    tm = tm.dropna()
    print('{} {}'.format('preproc',time()-start))
    # DETECT LANGUAGE
    if len(tm)>50:
        tm_sample = tm.sample(50)
    if len(tm)<50:
        tm_sample = tm
    tm_sample['srcdet'] = tm_sample['src'].apply(detect)
    tm_sample['trgdet'] = tm_sample['trg'].apply(detect)
    srcdet = cnt(tm_sample['srcdet']).most_common(1)[0][0]
    trgdet = cnt(tm_sample['trgdet']).most_common(1)[0][0]
    print('{} {}'.format('detect',time()-start))
    print('detected src lang: {} \ndetected trg lang: {}'.format(srcdet, trgdet))
    # SPLIT TOKENIZE
    print('{} {}'.format('get tokens',time()-start))
    tm['src'] = tm['src'].str.split()
    tm['trg'] = tm['trg'].str.split()
    # DROP EMPTY SEGMENTS
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    # GET STOPWORDS FROM DETECTED LANGUAGES
    src_stops = get_stopwords(srcdet)
    trg_stops = get_stopwords(trgdet)
    # GET NGRAMS. Get bi and trigrams from tokenized sentences.
    tm['src'] = tm['src'].apply(lambda s: get_grams(s, src_stops))
    tm['trg'] = tm['trg'].apply(lambda s: get_grams(s, trg_stops))      
    print('{} {}'.format('grams', time()-start))                    
    # REMOVE STOP WORDS. Remove stop words AFTER ngram extraction.  
    stops = src_stops + trg_stops
    tm['src'] = tm['src'].apply(lambda s: [w for w in s if not w.lower() in stops])
    tm['trg'] = tm['trg'].apply(lambda s: [w for w in s if not w.lower() in stops])
    print('{} {}'.format('remove stops', time()-start))  
    # DROP NON ALFANUMERIC TOKENS (replace-alpha check to consider space-separated ngrams)
    tm['src'] = tm['src'].apply(lambda s: [w for w in s if w.replace(' ','').isalpha()])
    tm['trg'] = tm['trg'].apply(lambda s: [w for w in s if w.replace(' ','').isalpha()])  
    # DROP EMPTY SEGMENTS
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    # MY gensim WORD VECTORIZER. Get gensim vector objects 
    tm, sd, td, stmd, ttmd = my_word_vectorizer(tm)
    # TM ITERZIP. Generate all possible source-target term combinations from segments
    tm = tm_iter_zip(tm)
    print('{} {}'.format('fn to iterzip', time()-start))
    # GET FREQ DICT OF PAIR CANDIDATES
    tbc = dict(cnt([pair for segment in tm['iter_zip'].tolist() for pair in segment]))
    # GET TB DF
    tb = pd.DataFrame(zip(tbc.keys(), tbc.values()))
    tb.columns = ['pair','pair_fq']
    # PAIR CANDIDATES ABOVE N
    tb = tb[tb['pair_fq']>1]
    # CREATE SRC AND TRG COLUMNS
    tb['src'] = tb['pair'].apply(lambda tup: tup[0])
    tb['trg'] = tb['pair'].apply(lambda tup: tup[1])
    # DROP SHORTS
    tb = tb[tb['src'].str.len()>3]
    tb = tb[tb['trg'].str.len()>3]
    # GET TERM FREQUENCY
    tb['src_fq'] = tb['src'].apply(lambda i: sd.cfs[sd.token2id[i]])
    tb['trg_fq'] = tb['trg'].apply(lambda i: td.cfs[td.token2id[i]])
    # GET TERM TFIDF
    tb['src_tfidf'] = tb['src'].apply(lambda i: stmd[sd.token2id[i]])
    tb['trg_tfidf'] = tb['trg'].apply(lambda i: ttmd[td.token2id[i]])
    tb = tb_prune_fqs(tb)
    # GET NON TRANSLATABLES TB
    tb_nt = tb[tb['src']==tb['trg']]
    # ADD 'ORIGIN' COLUMN
    tb_nt['origin'] = 'nt'
    # DROP NON TRANS FROM TB
    tb = tb[tb['src'].isin(tb_nt['src'])==False]
    tb = tb[tb['trg'].isin(tb_nt['trg'])==False]
    print('{} {}'.format('prepare {} tb cands'.format(len(tb)), time()-start))
    # SEND TB TO AZURE
    tb = tb_to_azure(tb, srcdet, trgdet)
    print('{} {}'.format('tb to azure',time()-start))
    # CLEAN MT RESULTS
    tb['mt_cands'] = tb['mt_cands'].apply(lambda l: [strip_punct(w) for w in l])
    # FIND IF ANY MT RESULT MATCHES TRG CANDIDATE
    tb['z'] = list(zip(tb['mt_cands'], tb['trg']))  
    # GET MT MATCHES
    tb['mt_match'] = tb['z'].apply(lambda t: get_mt_match(t))
    # DROP NAN AND HELPER COLUMNS
    tb = tb.dropna()
    tb = tb.drop(columns=['z','mt_match'])
    # ADD 'ORIGIN' COLUMN
    tb['origin'] = 'mt'
    # not found
    #x = [w for w in unique_src_terms if not w in tb['src'].tolist()]
    # CONCAT MT TB AND NT TB
    tb = pd.concat([tb_nt, tb])
    tb = tb.sort_values(by='pair_fq',ascending=False)
    tb = pd.DataFrame(zip(tb['src'],tb['trg']))
    tb.columns=['src','trg']
    tb = tb[tb['src'].str.len()>3]
    tb = tb[tb['trg'].str.len()>3]
    tb.to_csv('{}/{}_tb.csv'.format(upload_path, filename), encoding='utf8', index=False)
    tb.columns = ['src','trg']
    print('{} {}'.format('got tb', time()-start))
    # ANALYZE AND DETERMINE FINAL TB
    #drop shorts
    tmtb_html = tmtb_to_html(upload_path, filename, tm, tb)
    print('{} {}'.format('total', time()-start))
    return tmtb_html
    # except:
    #     return '<p>There was an error processing your file</p>'

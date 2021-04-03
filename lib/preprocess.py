#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing module. Drops empty segments, cleans html, makes lowercase,
cleans \n, \t, \r symbols, word tokenizes and removes stop words.
"""
import re
from bs4 import BeautifulSoup
from langdetect import detect
from collections import Counter as cnt
from lib.get_stopwords import get_stopwords
from lib.get_grams import get_grams

def preprocess(tm):
# PREPROCESS. Handle html, punctuation and other dirty characters.
    tm.columns = ['src','trg']
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    # KEEP MOSTLY ALPHA. Keep segments with > 85% alphabetic characters
    tm['src'] = tm['src'].apply(mostly_alpha)
    tm['trg'] = tm['trg'].apply(mostly_alpha)
    tm = tm.dropna()
    # PRESERVE ORIGINAL STRINGS
    tm['srcu'] = tm['src']
    tm['trgu'] = tm['trg']
    # CLEAN HTML
    tm['src'] = tm['src'].apply(html_check)
    tm['trg'] = tm['trg'].apply(html_check)
    # FIX CAMELCASE like this: HelloWorld -> Hello World
    tm['src'] = tm['src'].apply(pad_camels)
    tm['trg'] = tm['trg'].apply(pad_camels)
    # REPLACE newlines and tabs \n, \t, \r with space
    tm['src'] = tm['src'].apply(clean_special_symbols)
    tm['trg'] = tm['trg'].apply(clean_special_symbols)
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
    # DETECT LANGUAGE
    if len(tm)>50:
        tm_sample = tm.sample(50)
    if len(tm)<50:
        tm_sample = tm
    tm_sample['srcdet'] = tm_sample['src'].apply(detect)
    tm_sample['trgdet'] = tm_sample['trg'].apply(detect)
    srcdet = cnt(tm_sample['srcdet']).most_common(1)[0][0]
    trgdet = cnt(tm_sample['trgdet']).most_common(1)[0][0]
    # SPLIT TOKENIZE
    tm['src'] = tm['src'].str.split()
    tm['trg'] = tm['trg'].str.split()
    # DROP EMPTY SEGMENTS
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    # GET STOPWORDS FROM DETECTED LANGUAGES
    src_stops = get_stopwords(srcdet)
    trg_stops = get_stopwords(trgdet)
    # GET NGRAMS
    tm['src'] = tm['src'].apply(lambda s: get_grams(s, src_stops))
    tm['trg'] = tm['trg'].apply(lambda s: get_grams(s, trg_stops))      
    # REMOVE STOP WORDS. Remove stop words AFTER ngram extraction.  
    stops = src_stops + trg_stops
    tm['src'] = tm['src'].apply(lambda s: [w for w in s if not w.lower() in stops])
    tm['trg'] = tm['trg'].apply(lambda s: [w for w in s if not w.lower() in stops])
    # DROP NON ALFANUMERIC TOKENS (replace-alpha check to consider space-separated ngrams)
    tm['src'] = tm['src'].apply(lambda s: [w for w in s if w.replace(' ','').isalpha()])
    tm['trg'] = tm['trg'].apply(lambda s: [w for w in s if w.replace(' ','').isalpha()])  
    # DROP EMPTY SEGMENTS
    tm = tm[tm['src'].str.len()>0]
    tm = tm[tm['trg'].str.len()>0]
    return tm, srcdet, trgdet



def mostly_alpha(line):
    #alpha = [c for c in line if c.isalpha()]
    not_alpha = [c for c in line if not c.isalpha() and not c== ' ']
    line_len = len(line)
    #alfa_len = len(alpha)
    not_alfa_len = len(not_alpha)
    if line_len > 0:
        not_alfa_ratio = not_alfa_len/line_len
        if not_alfa_ratio < .15 and not 'http' in line:
            return line
        else:
            return None
    else:
        return None

def extract_text(string):
    blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head', 
    'input',
    'script',
    ]
    text = BeautifulSoup(string,'html.parser').find_all(text=True)
    output = ''
    for t in text:
        if t.parent.name not in blacklist:
            output += '{}'.format(t)
    return output

def html_check(string):
    if bool(BeautifulSoup(string, "html.parser").find()) == True:
        text = extract_text(string)
        return text
    else:
        return string

def pad_camels(string):
    try:
        pattern = '([a-z])([A-Z|||])'
        x = re.findall(pattern, string)
        for y in [''.join((a,b)) for (a,b) in x]:
            string = string.replace(y, '{} {}'.format(y[0],y[1]))
        return string
    except:
        return string

def pad_punct(string):
    """Fast punct padding function"""
    #pad all non-alphabetic characters
    new_string = re.sub("([^a-z|0-9|A-Z|\s|ñ|á|é|í|ó|ú|ü|ß|Ö|@|'|’])", r' \1 ', string)
    #replace double space with single space
    new_string = re.sub('\s{2,}', ' ', new_string)
    return new_string

def drop_double_spaces(string):
    new_string = re.sub('\s{2,}', ' ', string)
    return new_string

def clean_special_symbols(string):
    new_string = re.sub(r"(\\n|\\t|\\r|nsbp;)", " ", string)
    return new_string

def strip_punct(line):
    line = str(line)
    charset = set()
    for ch in line:
        charset.update(ch)
    punct = [ch for ch in charset if not ch.isalpha()]
    if ' ' in punct:
        punct.remove(' ')
    if "'" in punct:
        punct.remove("'")
    if "’" in punct:
        punct.remove("’")
    for ch in punct:
        line = line.replace(ch, ' ')
        line = drop_double_spaces(line)
        line = line.strip()
    return line


def correct_ellipsis(string):
    new_string = re.sub('(\.\.\.)', '…', string)
    return new_string
   

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 07:24:59 2021
@author: Luis Mondragon
Preprocessing module. Drops empty segments, cleans html, makes lowercase,
cleans \n, \t, \r symbols, word tokenizes and removes stop words.
"""
import re
from bs4 import BeautifulSoup

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

# def pad_punct(string):
#     return ''.join([c if c.isalpha() or c == ' ' else ' {} '.format(c) for c in string])

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
   


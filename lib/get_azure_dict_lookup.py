#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:30:28 2021

@author: user
"""
import requests, uuid
subscription_key = 'secret'

def get_normalizedTargets(d):
    targets = [t['normalizedTarget'] for t in d['translations']]
    if len(targets)>0:
        return targets
    else:
        return None

def get_azure_dict_lookup(src_lang, trg_lang, textinput):
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/dictionary/lookup?api-version=3.0'
    constructed_url = '{}{}&from={}&to={}'.format(base_url, path, src_lang, trg_lang)

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': 'westus2',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = []
    for w in textinput:
        body.append({'text': w})
        
    response = requests.post(constructed_url, headers=headers, json=body).json()
    targets = [get_normalizedTargets(d) for d in response]
    pairs = list(zip(textinput, targets))
    return pairs







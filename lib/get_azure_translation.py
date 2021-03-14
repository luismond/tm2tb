#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 02:34:27 2021

@author: user
"""

import requests, uuid
subscription_key = 'secret'

def get_translated_text(d):
    translations = [t['text'] for t in d['translations']]
    if len(translations)>0:
        return translations
    else:
        return None

def get_azure_translation(textinput, language_output):
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&to=' + language_output
    constructed_url = base_url + path + params

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
        
    response = requests.post(constructed_url, headers=headers, json=body)
    response_j = response.json()
    translations = [get_translated_text(t) for t in response_j]
    pairs = list(zip(textinput, translations))
    return pairs



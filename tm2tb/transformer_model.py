#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load transformer model
"""
from sentence_transformers import SentenceTransformer

class TransformerModel:
    def __init__(self):
        self.path = '/home/user/TM2TB/distiluse-base-multilingual-cased-v1'

    def load(self):
        print('Loading sentence transformer model...')
        model = SentenceTransformer(self.path)
        return model

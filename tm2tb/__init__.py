"""
TM2TB initialization
"""
__author__ = "Luis Mondragon (luismond@gmail.com)"
__version__ = "1.4.0"

from tm2tb.transformer_model import TransformerModel

trf_model = TransformerModel().load()
from tm2tb.spacy_models import get_spacy_model
from tm2tb.bitext_reader import BitextReader
from tm2tb.term_extractor import TermExtractor
from tm2tb.biterm_extractor import BitermExtractor

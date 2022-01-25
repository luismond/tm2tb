"""
TM2TB initialization
"""
__author__ = "Luis Mondragon (luismond@gmail.com)"
__version__ = '1.4.0'

from tm2tb.transformer_model import TransformerModel
trf_model = TransformerModel().load()
from tm2tb.bitext_reader import BitextReader
from tm2tb.sentence import Sentence
from tm2tb.bisentence import BiSentence
from tm2tb.text import Text
from tm2tb.bitext import BiText
from tm2tb.tm2tb import Tm2Tb

"""TM2TB initialization."""

__author__ = "Luis Mondragon (luismond@gmail.com)"
__version__ = "2.5"

from tm2tb.core.config import get_spacy_model, trf_model
from tm2tb.core.io_utils import BitextReader
from tm2tb.core.term_extractor import TermExtractor
from tm2tb.core.biterm_extractor import BitermExtractor

__all__ = ["TermExtractor", "BitermExtractor", "BitextReader", "trf_model"]

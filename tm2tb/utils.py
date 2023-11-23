from langdetect import detect, LangDetectException
from collections import Counter as cnt
from typing import Union
from random import randint
import re

def detect_lang(input_: Union[str, list]):
    """
    This method uses langdetect to detect the language of one or more sentences.
    (See https://pypi.org/project/langdetect/)
    The languages supported depends on the spaCy language modules installed.

    For one sentence, it passes the sentence to the langdetect module.
    For multiple sentences, it takes the most common language detected from a sample of sentences.
    This is done to speed-up the process.

    Parameters
    ----------
    input_ : Union[str, list]
        DESCRIPTION. String or list of strings

    Raises
    ------
    ValueError
        DESCRIPTION. Error raised when the language identified is not supported.

    Returns
    -------
    lang : string
        DESCRIPTION. Two-character language identifier.

    """
    supported_languages = ['en', 'es', 'de', 'fr', 'pt', 'it']
    if isinstance(input_, str):
        lang = detect(input_)
        if lang not in supported_languages:
            raise ValueError('Language not supported!')
    if isinstance(input_, list):
        text = input_
        if len(text)<=50:
            text_sample = text
        else:
            rand_start = randint(0, (len(text)-1)-50)
            text_sample = text[rand_start:rand_start+50]
        detections = []

        for sentence in text_sample:
            try:
                detections.append(detect(sentence))
            except LangDetectException:
                pass
        if len(detections)==0:
            raise ValueError('Insufficient data to detect language')
        lang = cnt(detections).most_common(1)[0][0]
        if lang not in supported_languages:
            raise ValueError('Language not supported!')
    return lang


def preprocess(sentence):
    """
    Minimal preprocessing function.
    It normalizes spaces and new line characters.

    Parameters
    ----------
    sentence : string
        DESCRIPTION. String representing one sentence or short paragraph.

    Returns
    -------
    string
        DESCRIPTION. The same string, but minimally cleaned.

    """

    def normalize_space_chars(sentence):
        """
        Replaces all spaces with normal spaces.
        """
        ords = [9, 10, 13, 32, 160]
        for char in sentence:
            if ord(char) in ords or char =='&nbsp;':
                sentence = sentence.replace(char, ' ')
        return sentence

    def normalize_space_seqs(sentence):
        """Finds sequences of more than one space, returns one space."""
        sentence = re.sub(r"(\s+)", ' ', sentence)
        return sentence

    def normalize_newline(sentence):
        """Replaces hard coded newlines with normal newline symbol."""
        def repl(sentence):
            groups = sentence.groups()
            return '{}{}{}'.format(groups[0], "\n", groups[2])
        pattern = r"(.)(\n|\\n|\\\n|\\\\n|\\\\\n)(.)"
        return re.sub(pattern, repl, sentence)

    return normalize_newline(normalize_space_seqs(normalize_space_chars(sentence)))

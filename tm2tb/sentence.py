"""
TM2TB Sentence class.
Implements methods for string cleaning, validation, tokenization, and ngram selection.
"""
import re
from langdetect import detect
from tm2tb.spacy_models import get_spacy_model

class Sentence:

    """
    A class to represent a sentence and its ngrams.

    Attributes
    ----------
    sentence : str
        Raw Unicode sentence, short text or paragraph.

    lang : str
        Detected language of the sentence.

    clean_sentence : str
        Preprocessed and cleaned sentence.

    supported_languages : list
        List of supported languages.

    Methods
    -------
    preprocess()
        Cleans and validates the sentence.

    get_ngrams(ngrams_min=, ngrams_max=, include_pos=, exclude_pos=)
        Gets ngrams candidates from the sentence
    """
    supported_languages = ['en', 'es', 'de', 'fr']

    def __init__(self, sentence):
        self.sentence = sentence
        self.lang = detect(self.sentence)
        self.clean_sentence = self.preprocess()

    def preprocess(self,
                   min_non_alpha_ratio = .25,
                   sentence_min_length = 30,
                   sentence_max_length = 700):
        """
        Normalizes spaces, apostrophes and special characters.
        Validates sentence alphabetic-ratio, length, and language.

        Parameters
        ----------
        min_non_alpha_ratio : float, optional
            DESCRIPTION. Minimum alphabetical characters ratio of sentence.
        sentence_min_length : int, optional
            DESCRIPTION. Sentence minimum length.
        sentence_max_length : int, optional
            DESCRIPTION. Sentence maximum length.


        Returns
        -------
        str
            String representing a preprocessed sentence.

        """

        def normalize_space_chars(sentence):
            """
            Replaces all spaces with normal spaces.
            """
            ords = [9, 10, 13, 32, 160]
            for char in sentence:
                if ord(char) in ords:
                    sentence = sentence.replace(char, ' ')
            return sentence

        def normalize_space_seqs(sentence):
            """
            Finds sequences of more than one space, returns one space.
            """
            # def repl(match):
            #     return ' '
            sentence = re.sub(r"(\s+)", ' ', sentence)
            return sentence

        def normalize_apostrophe(sentence):
            """
            Replaces curved apostrophe with straight apostrophe.
            """
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"'s", groups[2])
            pattern = r"(.|\s)(’s)(.|\s)"
            return re.sub(pattern, repl, sentence)

        def normalize_newline(sentence):
            """
            Replaces hard coded newlines with normal newline symbol.
            """
            def repl(sentence):
                groups = sentence.groups()
                return '{}{}{}'.format(groups[0],"\n", groups[2])
            pattern = r"(.)(\n|\\n|\\\n|\\\\n|\\\\\n)(.)"
            return re.sub(pattern, repl, sentence)

        def validate_if_mostly_alphabetic(sentence):
            """
            Checks if most of the characters in sentence are alphabetic.
            """
            alpha = len([char for char in sentence if char.isalpha()])
            if alpha==0:
                raise ValueError('No alphanumeric chars found!')
            non_alpha = len([char for char in sentence
                             if not char.isalpha() and not char==' '])
            non_alpha_ratio = non_alpha/alpha
            if non_alpha_ratio >= min_non_alpha_ratio:
                raise ValueError('Too many non-alpha chars!')
            if sentence.startswith('http'):
                raise ValueError('Cannot process http addresses!')
            if sentence.isdigit():
                raise ValueError('Sentence contains only numbers!')
            return sentence

        def validate_length(sentence):
            """
            Checks if sentence length is between min and max length values.
            """
            if len(sentence) <= sentence_min_length:
                raise ValueError('Sentence is too short!')
            if len(sentence) >= sentence_max_length:
                raise ValueError('Sentence is too long!')
            return sentence

        def validate_lang(sentence):
            """
            Checks if sentence language is supported.
            """
            if self.lang not in self.supported_languages:
                raise ValueError('Language not supported!')
            return sentence

        sentence = normalize_space_chars(self.sentence)
        sentence = normalize_space_seqs(sentence)
        sentence = normalize_apostrophe(sentence)
        sentence = normalize_newline(sentence)
        sentence = validate_if_mostly_alphabetic(sentence)
        sentence = validate_length(sentence)
        sentence = validate_lang(sentence)
        return sentence

    def generate_ngrams(self,
                   ngrams_min = 1,
                   ngrams_max = 2):
        "Generate ngrams from sentence sequence"

        # Get spacy model and instantiate a spacy doc with the clean sentence
        spacy_model = get_spacy_model(self.lang)
        doc = spacy_model(self.clean_sentence)

        # Get text and part-of-speech tag for each token in document
        pos_tokens = [(token.text, token.pos_) for token in doc]

        # Get n-grams from pos_tokens
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def get_candidate_ngrams(self,
                      include_pos = None,
                      exclude_pos = None,
                      **kwargs):
        """
        Filter ngrams with part-of-speech tags and punctuation rules.

        Parameters
        ----------

        include_pos : list
            DESCRIPTION.    A list of POS-tags to delimit the ngrams.
                            If None, the default value is ['NOUN', 'PROPN']

        exclude_pos : list
            DESCRIPTION.    A list of POS-tags to exclude from the ngrams.

        **kwargs : dict

            See below

            Optional Keyword Arguments:
                ngrams_min : int, optional
                    DESCRIPTION.    Minimum ngram sequence length.
                                    The default value is 1.

                ngrams_max : int, optional
                    DESCRIPTION.    Maximum ngram sequence length.
                                    The default value is 2.

        Returns
        -------
        dict
            DESCRIPTION. Dictionary of n-grams and part-of-speech tags

        """


        pos_ngrams = self.generate_ngrams(**kwargs)

        #include_punct = ["'", ":", "’", "’", "'", "™", "®", "%"]
        exclude_punct = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
                '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']

        if include_pos is None:
            include_pos = ['NOUN','PROPN']
        if exclude_pos is None:
            exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX', 'VERB']

        # Keep ngrams where the first element's pos-tag
        # and the last element's pos-tag are present in include_pos
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][1] in include_pos
                          and pos_ngram[-1:][0][1] in include_pos, pos_ngrams)

        # Keep ngrams where none of elements' tag is in exclude pos
        pos_ngrams = filter(lambda pos_ngram: not any(token[1] in exclude_pos
                                                      for token in pos_ngram), pos_ngrams)

        # Keep ngrams where the first element's token
        # and the last element's token are alpha
        pos_ngrams = filter(lambda pos_ngram: pos_ngram[0][0].isalpha()
                          and pos_ngram[-1:][0][0].isalpha(), pos_ngrams)

        # Keep ngrams where none of the middle elements' text is in exclude punct
        pos_ngrams = filter(lambda pos_ngram: not any((token[0] in exclude_punct
                                                       for token in pos_ngram[1:-1])), pos_ngrams)

        def rejoin_special_punct(ngram):
            'Joins apostrophes and other special characters to their token.'
            def repl(match):
                groups = match.groups()
                return '{}{}{}'.format(groups[0],groups[2], groups[3])
            pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
            return re.sub(pattern, repl, ngram)

        pos_ngrams_ = {'ngrams':[],
                       'joined_ngrams':[],
                       'tags':[]}

        for pos_ngram in pos_ngrams:
            ngram, tag = zip(*pos_ngram)
            joined_ngram = rejoin_special_punct(' '.join(ngram))
            pos_ngrams_['ngrams'].append(ngram)
            pos_ngrams_['joined_ngrams'].append(joined_ngram)
            pos_ngrams_['tags'].append(tag)

        return list(set(pos_ngrams_['joined_ngrams']))#move this to next method

import re

def preprocess(sentence,
               min_non_alpha_ratio = .25,
               sentence_min_length = 5,
               sentence_max_length = 3000):
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

    sentence = normalize_space_chars(sentence)
    sentence = normalize_space_seqs(sentence)
    sentence = normalize_apostrophe(sentence)
    sentence = normalize_newline(sentence)
    sentence = validate_if_mostly_alphabetic(sentence)
    sentence = validate_length(sentence)
    return sentence
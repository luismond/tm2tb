from collections import Counter as cnt
import re
import pandas as pd

def filter_ngrams(pos_tokens,
                  ngrams_min=1,
                  ngrams_max=3,
                  min_freq=2,
                  include_pos=None,
                  exclude_pos=None):

    pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
              for n in range(ngrams_min, ngrams_max+1))
    pos_ngrams = (ng for ngl in pos_ngrams for ng in ngl)
    pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=min_freq]

    exclude_punct = [',','.','/','\\','(',')','[',']','{','}',';','|','"','!',
            '?','…','...', '<','>','“','”','（','„',"'",',',"‘",'=','+']

    if include_pos is None:
        include_pos = ['NOUN', 'PROPN', 'ADJ']
    if exclude_pos is None:
        exclude_pos = ['X', 'SCONJ', 'CCONJ', 'AUX', 'VERB']
        exclude_pos = [tag for tag in exclude_pos if not tag in include_pos]
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

    # check if POS n-grams are empty
    pos_ngrams = [list(pn) for pn in pos_ngrams]
    if len(pos_ngrams)==0:
        raise ValueError('No POS n-grams left after filtering!')

    def rejoin_special_punct(ngram):
        'Joins apostrophes and other special characters to their token.'
        def repl(match):
            groups = match.groups()
            return '{}{}{}'.format(groups[0],groups[2], groups[3])
        pattern = r"(.+)(\s)('s|:|’s|’|'|™|®|%)(.+)"
        return re.sub(pattern, repl, ngram)

    # Make data frame from n-grams and parts-of-speech
    pos_ngrams_ = pd.DataFrame([zip(*pos_ngram) for pos_ngram in pos_ngrams])
    pos_ngrams_.columns = ['ngrams','tags']
    pos_ngrams_.loc[:, 'joined_ngrams'] = \
        pos_ngrams_['ngrams'].apply(lambda ng: rejoin_special_punct(' '.join(ng)))
    pos_ngrams_ = pos_ngrams_.drop_duplicates(subset='joined_ngrams')
    pos_ngrams_ = pos_ngrams_.reset_index()
    pos_ngrams_ = pos_ngrams_.drop(columns=['index'])
    return pos_ngrams_

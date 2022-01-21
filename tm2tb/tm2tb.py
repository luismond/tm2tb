"""
TM2TB module. Extracts terms from sentences, pairs of sentences and bilingual documents.
@author: Luis Mondragon (luismond@gmail.com)
Last updated on Tue Jan 21 04:55:22 2022
"""
import re
from collections import Counter as cnt
from typing import Union, Tuple, List
import numpy as np
import pandas as pd
from random import randint

from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect

from tm2tb.spacy_models import get_spacy_model
from tm2tb import BitextReader
from tm2tb.preprocess import preprocess

pd.options.mode.chained_assignment = None

print('Loading LaBSE model...')
model = SentenceTransformer('')

class Tm2Tb:
    """
    A class that represents a bilingual n-gram extractor.

    Attributes
    ----------
    model: sentence_transformers.SentenceTransformer.SentenceTransformer
        LaBSE sentence transformer model
        https://huggingface.co/sentence-transformers/LaBSE

    Methods
    -------
    get_ngrams()
        Gets ngrams from sentences, tuples of sentences or list of tuples of sentences.
    """

    def __init__(self, model=model):
        self.model = model

    def get_ngrams(self, input_: Union[str, Tuple[tuple], List[list]], fast=False, **kwargs):
        """
        Parameters
        ----------
        input_ : Union[str, Tuple[tuple], List[list]]
            str:    A string representing a sentence (or short paragraph).

            tuple:  A tuple of two sentences.
                    (For example, a sentence in English and its translation to Spanish).

            list:   A list of tuples of two sentences.

        **kwargs : dict

        See below

        Optional Keyword Arguments:
            ngrams_min : int, optional
                DESCRIPTION.    Minimum ngram sequence length.
                                The default value is 1.

            ngrams_max : int, optional
                DESCRIPTION.    Maximum ngram sequence length.
                                The default value is 2.

            include_pos : list, optional
                DESCRIPTION.    A list of POS-tags to delimit the ngrams.
                                If None, the default value is ['NOUN', 'PROPN']

            exclude_pos : list, optional
                DESCRIPTION.    A list of POS-tags to exclude from the ngrams.

            top_n : int, optional
                DESCRIPTION.    An integer representing the maximum number of results
                                of single sentence ngrams.
                                The default value is len(candidate_ngrams)*.5

            diversity : float, optional
                DESCRIPTION.    A float representing the diversity of single-sentence ngrams.
                                It is used to calculate Maximal Marginal Relevance.
                                The default value is 0.5

            min_similarity : float, optional
                DESCRIPTION.    A float representing the minimum similarity allowed between
                                ngrams from the source sentence & ngrams from the target sentence.
                                The default value is .8

        Returns
        -------
        ngrams : Pandas Data Frame 
            DESCRIPTION. Data Frame representing the ngrams and their metadata
        """
        
        if isinstance(input_, str):
            ngrams = Sentence(input_).get_top_ngrams(**kwargs)

        elif isinstance(input_, tuple):
            src_sentence, trg_sentence = input_
            ngrams = BiSentence(src_sentence, trg_sentence).get_top_ngrams(**kwargs)
       
        elif isinstance(input_, list):
            bitext = input_
            if fast is False:
                ngrams = BiText(bitext).get_top_ngrams(**kwargs)
            if fast is True:
                ngrams = BiText(bitext).get_top_ngrams_fast()

        return ngrams
    
    
    @classmethod
    def read_bitext(cls, file_path):
        """
        Parameters
        ----------
        file_path : str
            String representing the full path of a bilingual file.
            See supported file formats in tm2tb.bilingual_reader.BilingualReader

        Returns
        -------
        bitext : list
            DESCRIPTION. A list of tuples of sentences.
                          It is assumed that they are translations of each other.
        """

        # Pass the path and file name to BilingualReader
        bitext = BitextReader(file_path).read_bitext()
        bitext = bitext.drop_duplicates(subset='src')
        bitext = bitext.drop_duplicates(subset='trg')
        bitext = list(zip(bitext['src'], bitext['trg']))
        return bitext

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

    generate_ngrams(ngrams_min=, ngrams_max=)
        Generate ngrams from the sentence within ngrams_min and ngrams_max range

    get_candidate_ngrams(include_pos=, exclude_pos=)
        Filters generated n-grams using part-of-speech tags and string rules
        
    get_best_sentence_ngrams(self, top_n = 30, diversity=.5, return_embs=False, **kwargs)
        Embeds sentence and candidate ngrams, calculates n-gram-to-sentence similarities
    """
    

    def _generate_ngrams(self, ngrams_min = 1, ngrams_max = 2):
        """
        Generate ngrams from sentence sequence
        """

        # Get spaCy model and instantiate a doc with the clean sentence
        spacy_model = get_spacy_model(self.lang)
       
        doc = spacy_model(self.clean_sentence)

        # Get text and part-of-speech tag for each token in document
        pos_tokens = [(token.text, token.pos_) for token in doc]

        # Get n-grams from pos_tokens
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def _get_candidate_ngrams(self, include_pos = None, exclude_pos = None, **kwargs):
        """
        Filter ngrams with part-of-speech tags and punctuation rules.

        Parameters
        ----------

        include_pos : list
            DESCRIPTION.    A list of POS-tags to delimit the ngrams.
                            If None, the default value is ['NOUN', 'PROPN', 'ADJ']

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
            DESCRIPTION. Data frame representing ngrams and part-of-speech tags
        """

        pos_ngrams = self._generate_ngrams(**kwargs)

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

    def get_top_ngrams(self, top_n = None, diversity=.8, return_embs=False, **kwargs):
        """
        Embed sentence and candidate ngrams.
        Calculate the best sentence ngrams using cosine similarity and MMR.
        """
        cand_ngrams_df = self._get_candidate_ngrams(**kwargs)
        joined_ngrams = cand_ngrams_df['joined_ngrams']
        
        if top_n is None:
            top_n = round(len(joined_ngrams)*.75)
        
        # Embed clean sentence and joined ngrams
        seq1_embeddings = model.encode([self.clean_sentence])
        seq2_embeddings = model.encode(joined_ngrams)

        # Get sentence/ngrams similarities
        ngram_sentence_sims = cosine_similarity(seq2_embeddings, seq1_embeddings)

        # Get ngrams/ngrams similarities
        ngram_sims = cosine_similarity(seq2_embeddings)

        # Initialize candidates and choose best ngram
        best_ngrams_idx = [np.argmax(ngram_sentence_sims)]

        # All ngrams that are not in best ngrams
        candidates_idx = [i for i in range(len(joined_ngrams)) if i != best_ngrams_idx[0]]
        

        for _ in range(min(top_n - 1, len(joined_ngrams) - 1)):
            # Get distances within candidates and between candidates and selected ngrams
            candidate_sims = ngram_sentence_sims[candidates_idx, :]
            rest_ngrams_sims = np.max(ngram_sims[candidates_idx][:, best_ngrams_idx], axis=1)

            # Calculate Maximum Marginal Relevance
            mmr = (1-diversity) * candidate_sims - diversity * rest_ngrams_sims.reshape(-1, 1)

            # Get closest candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update best ngrams & candidates
            best_ngrams_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
            
        # Keep only ngrams in best_ngrams_idx
        best_ngrams_df = cand_ngrams_df.iloc[best_ngrams_idx]

        # Add rank and embeddings
        best_ngrams_df.loc[:, 'rank'] = [round(float(ngram_sentence_sims.reshape(1, -1)[0][idx]), 4)
                                    for idx in best_ngrams_idx]
        best_ngrams_df.loc[:, 'embedding'] = [seq2_embeddings[idx] for idx in best_ngrams_idx]
        best_ngrams_df = best_ngrams_df.sort_values(by='rank', ascending = False)
        if return_embs is False:
            best_ngrams_df = best_ngrams_df.drop(columns=['ngrams','tags','embedding'])
        return best_ngrams_df

class BiSentence:
    def __init__(self, src_sentence, trg_sentence):
        self.src_sentence = src_sentence
        self.trg_sentence = trg_sentence
        
    def get_top_ngrams(self, min_similarity=.85, **kwargs):

        src_ngrams_df = Sentence(self.src_sentence).get_top_ngrams(return_embs=True, **kwargs)
        trg_ngrams_df = Sentence(self.trg_sentence).get_top_ngrams(return_embs=True, **kwargs)
     
        seq_similarities = cosine_similarity(src_ngrams_df['embedding'].tolist(),
                                              trg_ngrams_df['embedding'].tolist())
        # get seq1 & seq2 indexes
        seq1_idx = list(range(len(src_ngrams_df['joined_ngrams'].tolist())))
        seq2_idx = list(range(len(trg_ngrams_df['joined_ngrams'].tolist())))

        # # get max seq2 values and indexes
        max_seq2_values = np.max(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
        max_seq2_idx = np.argmax(seq_similarities[seq1_idx][:, seq2_idx], axis=1)

        # make bi_ngrams data frame with the top src_ngram/trg_ngram similarities
        bi_ngrams = pd.DataFrame([(src_ngrams_df.iloc[idx]['joined_ngrams'],
                                   src_ngrams_df.iloc[idx]['rank'],
                                   src_ngrams_df.iloc[idx]['tags'],
                                   trg_ngrams_df.iloc[max_seq2_idx[idx]]['joined_ngrams'],
                                   trg_ngrams_df.iloc[max_seq2_idx[idx]]['rank'],
                                   trg_ngrams_df.iloc[max_seq2_idx[idx]]['tags'],
                                   float(max_seq2_values[idx])) for idx in seq1_idx])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_rank',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_rank',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']

        # # Keep n-grams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= min_similarity]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')

        # Group by source, get the most similar target n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (src_ngram, df) in list(bi_ngrams.groupby('src_ngram'))])

        # Group by target, get the most similar source n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (trg_ngram, df) in list(bi_ngrams.groupby('trg_ngram'))])

        # Get bi n-gram rank
        bi_ngrams['bi_ngram_rank'] = bi_ngrams['bi_ngram_similarity'] * \
            bi_ngrams['src_ngram_rank'] * bi_ngrams['trg_ngram_rank']
        bi_ngrams = bi_ngrams.sort_values(by='bi_ngram_rank', ascending=False)

        # Finish
        bi_ngrams = bi_ngrams.round(4)
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        return bi_ngrams


class BiText:
    "Implements fast methods for bilingual terminology extraction"
    "Bitext is a list of tuples of sentences"
    def __init__(self, bitext):
        self.bitext = bitext
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sample_len = 20
        self.src_lang, self.trg_lang = self.get_bitext_langs()

    def get_bitext_sample(self):
        rand_start = randint(0,(len(self.bitext)-1)-self.sample_len)
        bitext_sample = self.bitext[rand_start:rand_start+self.sample_len]
        return bitext_sample

    def get_bitext_langs(self):
        if len(self.bitext)<=self.sample_len:
            bitext_sample = self.bitext
        else:
            bitext_sample = self.get_bitext_sample()
        src_lang = detect(' '.join(i[0] for i in bitext_sample))
        trg_lang = detect(' '.join(i[1] for i in bitext_sample))
        if src_lang not in self.supported_languages or\
            trg_lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return src_lang, trg_lang

    def preproc_bitext_sentences(self):
        src_sentences = []
        trg_sentences = []
        for i, _ in enumerate(self.bitext):
            try:
                src_sentence_raw = self.bitext[i][0]
                src_sentence = preprocess(src_sentence_raw)
                src_sentences.append(src_sentence)
                trg_sentence_raw = self.bitext[i][1]
                trg_sentence = preprocess(trg_sentence_raw)
                trg_sentences.append(trg_sentence)
            except:
                pass
        if len(src_sentences) == 0 or len(trg_sentences) == 0:
            raise ValueError('No clean sentences left!')
        return src_sentences, trg_sentences

    def get_bitext_pos_tagged_tokens(self):
        src_sentences, trg_sentences = self.preproc_bitext_sentences()
        src_model = get_spacy_model(self.src_lang)
        sdocs = list(src_model.pipe(src_sentences))
        sc_doc = Doc.from_docs(sdocs)
        src_pos_tagged_tokens = [(token.text, token.pos_) for token in sc_doc]
        trg_model = get_spacy_model(self.trg_lang)
        tdocs = list(trg_model.pipe(trg_sentences))
        tc_doc = Doc.from_docs(tdocs, ensure_whitespace=True)
        trg_pos_tagged_tokens = [(token.text, token.pos_) for token in tc_doc]
        return src_pos_tagged_tokens, trg_pos_tagged_tokens

    @staticmethod
    def get_ngrams(pos_tokens):
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(1, 4+1))
        return (ng for ngl in pos_ngrams for ng in ngl)

    def generate_bitext_ngrams(self):
        src_pos_tagged_tokens, trg_pos_tagged_tokens = self.get_bitext_pos_tagged_tokens()
        src_pos_tagged_ngrams = self.get_ngrams(src_pos_tagged_tokens)
        trg_pos_tagged_ngrams = self.get_ngrams(trg_pos_tagged_tokens)
        return src_pos_tagged_ngrams, trg_pos_tagged_ngrams

    @staticmethod
    def get_candidate_ngrams(pos_ngrams, include_pos = None, exclude_pos = None):
        freq_min = 2
        pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=freq_min]

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

    def get_bitext_candidate_ngrams(self):
        src_pos_tagged_ngrams, trg_pos_tagged_ngrams = self.generate_bitext_ngrams()
        src_candidate_ngrams = self.get_candidate_ngrams(src_pos_tagged_ngrams)
        trg_candidate_ngrams = self.get_candidate_ngrams(trg_pos_tagged_ngrams)
        return src_candidate_ngrams, trg_candidate_ngrams

    @staticmethod
    def get_bitext_top_ngrams_partial(ngrams0, ngrams1):
        # Get top bitext ngrams from one side
        src_ngrams = ngrams0['joined_ngrams'].tolist()
        trg_ngrams = ngrams1['joined_ngrams'].tolist()
        # Get POS tags
        src_tags = ngrams0['tags'].tolist()
        trg_tags = ngrams1['tags'].tolist()
        # Get embeddings
        src_embeddings = model.encode(src_ngrams)
        trg_embeddings = model.encode(trg_ngrams)
        # Get similarities
        candidate_similarities = cosine_similarity(src_embeddings, trg_embeddings)
        # Get indexes
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
        # Get max trg ngrams values and indexes
        max_trg_values = np.max(candidate_similarities[src_idx][:, trg_idx], axis=1)
        max_trg_idx = np.argmax(candidate_similarities[src_idx][:, trg_idx], axis=1)
        # make ngrams dataframe with the top src_ngram/trg_ngram similarities
        bi_ngrams = pd.DataFrame([(src_ngrams[idx],
                                   src_tags[idx],
                                   trg_ngrams[max_trg_idx[idx]],
                                   trg_tags[max_trg_idx[idx]],
                                   float(max_trg_values[idx])) for idx in src_idx])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']
        # # Keep ngrams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= .85]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        # Finish
        bi_ngrams = bi_ngrams.round(4)
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        # print('finnish')
        # print(time()-START)
        return bi_ngrams

    def get_top_ngrams_fast(self):
        """
        Extract and filter all source ngrams and all target ngrams. 
        Find their most similar matches.
        Much faster, less precise, can cause OOM errors.
        """
        src_ngrams, trg_ngrams = self.get_bitext_candidate_ngrams()
        # Get top bitext ngrams from the source side
        bi_ngramss = self.get_bitext_top_ngrams_partial(src_ngrams, trg_ngrams)
        # Get top bitext ngrams from the target side
        bi_ngramst = self.get_bitext_top_ngrams_partial(trg_ngrams, src_ngrams)
        # Rearrange columns
        bi_ngramst.columns = ['trg_ngram',
                             'trg_ngram_tags',
                             'src_ngram',
                             'src_ngram_tags',
                             'bi_ngram_similarity']
        # Concat results
        bi_ngrams = pd.concat([bi_ngramss, bi_ngramst])
        # Drop duplicates
        bi_ngrams['st'] = [''.join(t) for t in list(zip(bi_ngrams['src_ngram'],
                                                        bi_ngrams['trg_ngram']))]
        bi_ngrams = bi_ngrams.drop_duplicates(subset='st')
        bi_ngrams = bi_ngrams.drop(columns=['st'])
        return bi_ngrams
    
    def get_top_ngrams(self, **kwargs):
        """
        Extract all bi-ngrams from all sentence tuples in bitext.
        Slow, more precise, less RAM-hungry.
        """
        bitext = self.bitext
        bi_ngrams = []
        for i, _ in enumerate(bitext):
            try:
                src_sentence = bitext[i][0]
                trg_sentence = bitext[i][1]
                bi_ngrams_ = BiSentence(src_sentence, trg_sentence).get_ngrams(**kwargs)
                # bi_ngrams_ = self._get_bi_ngrams_from_bisentence(src_sentence,
                #                                                  trg_sentence,
                #                                                  **kwargs)
                bi_ngrams.append(bi_ngrams_)
            except:
                pass

        if len(bi_ngrams) == 0:
            raise ValueError('No bilingual n-gram candidates found!')

        # Concatenate sentence-level n-grams
        bi_ngrams = pd.concat(bi_ngrams)
        bi_ngrams = bi_ngrams.reset_index()

        # Get rank avgs
        src_avg = {a: round(b['src_ngram_rank'].sum()/len(b), 4)
                    for (a, b) in list(bi_ngrams.groupby('src_ngram'))}
        trg_avg = {a: round(b['trg_ngram_rank'].sum()/len(b), 4)
                    for (a, b) in list(bi_ngrams.groupby('trg_ngram'))}

        # Get frequencies
        src_counts = cnt(bi_ngrams['src_ngram'])
        trg_counts = cnt(bi_ngrams['trg_ngram'])

        # Drop duplicates
        bi_ngrams = bi_ngrams.drop_duplicates(subset='src_ngram')
        bi_ngrams = bi_ngrams.drop_duplicates(subset='trg_ngram')

        # Apply rank avgs
        bi_ngrams['src_ngram_rank'] = \
            bi_ngrams['src_ngram'].apply(lambda ngram: src_avg[ngram])
        bi_ngrams['trg_ngram_rank'] = \
            bi_ngrams['trg_ngram'].apply(lambda ngram: trg_avg[ngram])

        # Apply frequencies
        bi_ngrams['src_ngram_freq'] = bi_ngrams['src_ngram'].apply(lambda s: src_counts[s])
        bi_ngrams['trg_ngram_freq'] = bi_ngrams['trg_ngram'].apply(lambda s: trg_counts[s])

        # Get bi-ngram rank
        bi_ngrams['bi_ngram_rank'] = bi_ngrams['bi_ngram_similarity'] * \
            bi_ngrams['src_ngram_rank'] * bi_ngrams['trg_ngram_rank']
        bi_ngrams = bi_ngrams.sort_values(by='bi_ngram_rank', ascending=False)

        # Finish
        bi_ngrams = bi_ngrams.round(4)
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])

        # Re-order columns
        columns = ['src_ngram', 'src_ngram_tags', 'src_ngram_freq', 'src_ngram_rank',\
                   'trg_ngram', 'trg_ngram_tags', 'trg_ngram_freq', 'trg_ngram_rank',\
                       'bi_ngram_similarity', 'bi_ngram_rank']

        bi_ngrams = bi_ngrams.reindex(columns=columns)
        return bi_ngrams

"""
TM2TB module. Extracts terms from sentences, pairs of sentences and bilingual documents.
@author: Luis Mondragon (luismond@gmail.com)
Last updated on Tue Jan 23 11:29:22 2022
"""
from collections import Counter as cnt
from typing import Union, Tuple, List
import numpy as np
import pandas as pd
from random import randint

from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException

from tm2tb.spacy_models import get_spacy_model
from tm2tb import BitextReader
from tm2tb.preprocess import preprocess
from tm2tb.filter_ngrams import filter_ngrams

pd.options.mode.chained_assignment = None

print('Loading sentence transformer model...')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

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

    def get_ngrams(self, 
                   input_: Union[str, Tuple[tuple], List[list]],
                   **kwargs):
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
            ngrams = BiSentence(input_).get_top_ngrams(**kwargs)
        elif isinstance(input_, list):
            ngrams = BiText(input_).get_top_ngrams(**kwargs)
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
    def __init__(self, sentence):
        self.sentence = sentence
        self.supported_languages = ['en', 'es']#, 'de', 'fr']
        self.clean_sentence = preprocess(self.sentence)
        self.lang = self.validate_lang()
        
    def validate_lang(self):
        lang = detect(self.clean_sentence)
        if not lang in self.supported_languages:
            raise ValueError('Language not supported!')
        else:
            return lang

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
        pos_ngrams = filter_ngrams(pos_ngrams, include_pos, exclude_pos)
        return pos_ngrams

    def get_top_ngrams(self, top_n = None, diversity=.8, return_embs=False, **kwargs):
        """
        Embed sentence and candidate ngrams.
        Calculate the best sentence ngrams using cosine similarity and MMR.
        """
        cand_ngrams_df = self._get_candidate_ngrams(**kwargs)
        joined_ngrams = cand_ngrams_df['joined_ngrams']
        
        if top_n is None:
            top_n = round(len(joined_ngrams)*.85)
        
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
    def __init__(self, sentence_tuple):
        self.src_sentence, self.trg_sentence = sentence_tuple
        
    def get_src_trg_top_ngrams(self, **kwargs):
        # Get source ngram dataframe
        src_ngrams_df = Sentence(self.src_sentence).get_top_ngrams(return_embs=True, **kwargs)
        # Get target ngram dataframe
        trg_ngrams_df = Sentence(self.trg_sentence).get_top_ngrams(return_embs=True, **kwargs)
        return src_ngrams_df, trg_ngrams_df
    
    @staticmethod
    def get_seq_similarities(src_ngrams_df, trg_ngrams_df):
        # Get source/target ngram similarity matrix
        src_seq_similarities = cosine_similarity(src_ngrams_df['embedding'].tolist(),
                                              trg_ngrams_df['embedding'].tolist())
        
        # Get target/source ngram similarity matrix
        trg_seq_similarities = cosine_similarity(trg_ngrams_df['embedding'].tolist(),
                                              src_ngrams_df['embedding'].tolist())
        return src_seq_similarities, trg_seq_similarities
    
    def get_aligned_ngrams(self, **kwargs):
        # Get ngrams, pos_tags and ranks from source & target sentences
        src_ngrams_df, trg_ngrams_df = self.get_src_trg_top_ngrams(**kwargs)
        src_seq_similarities, trg_seq_similarities = self.get_seq_similarities(src_ngrams_df,
                                                                               trg_ngrams_df)
        
        src_ngrams = src_ngrams_df['joined_ngrams'].tolist()
        trg_ngrams = trg_ngrams_df['joined_ngrams'].tolist()
        src_tags = src_ngrams_df['tags'].tolist()
        trg_tags = trg_ngrams_df['tags'].tolist()
        src_ranks = src_ngrams_df['rank'].tolist()
        trg_ranks = trg_ngrams_df['rank'].tolist()
    
        # Get source ngram & target ngram indexes 
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
        
        # Get indexes and values of most similar target ngram for each source ngram
        src_max_values = np.max(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        src_max_idx = np.argmax(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        
        # Get indexes and values of most similar source ngram for each target ngram
        trg_max_values = np.max(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        trg_max_idx = np.argmax(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        
        # Align target ngrams & metadata with source ngrams & metadata
        src_aligned_ngrams = pd.DataFrame([(src_ngrams[idx],
                                            src_ranks[idx],
                                            src_tags[idx],
                                            trg_ngrams[trg_max_idx[idx]],
                                            trg_ranks[trg_max_idx[idx]],
                                            trg_tags[trg_max_idx[idx]],
                                            float(trg_max_values[idx])) for idx in src_idx])
        
        # Align source ngrams & metadata with target ngrams & metadata
        trg_aligned_ngrams = pd.DataFrame([(src_ngrams[src_max_idx[idx]],
                                            src_ranks[src_max_idx[idx]],
                                            src_tags[src_max_idx[idx]],
                                            trg_ngrams[idx],
                                            trg_ranks[idx],
                                            trg_tags[idx],
                                            float(src_max_values[idx])) for idx in trg_idx])
        return src_aligned_ngrams, trg_aligned_ngrams

    def get_top_ngrams(self, min_similarity=.8, **kwargs):
        # Concatenate source & target ngram alignments   
        src_aligned_ngrams, trg_aligned_ngrams = self.get_aligned_ngrams(**kwargs)
        bi_ngrams = pd.concat([src_aligned_ngrams, trg_aligned_ngrams])
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_rank',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_rank',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']
        
        # Keep n-grams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= min_similarity]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        
        # For one-word terms, keep those longer than 1 character
        bi_ngrams = bi_ngrams[bi_ngrams['src_ngram'].str.len()>1]
        bi_ngrams = bi_ngrams[bi_ngrams['trg_ngram'].str.len()>1]
        
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
        bi_ngrams = bi_ngrams.round(4)
        return bi_ngrams


class Text:
    """
    Implements fast methods for monolingual terminology extraction.
    Text is a list of sentences.
    """
    def __init__(self, text):
        self.text = text
        self.supported_languages = ['en', 'es', 'de', 'fr']
        self.sentences_clean = self.preprocess_sentences()
        self.lang = self.detect_text_lang()
        self.candidate_ngrams = self.get_candidate_ngrams()
    
    def preprocess_sentences(self):
        """
        Batch preprocess sentences
        """
        sentences = []
        for sentence in self.text:
            try:
                sentences.append(preprocess(sentence))
            except:
                pass
        if len(sentences)==0:
            raise ValueError('No clean sentences!')
        return sentences
    
    def detect_text_lang(self):
        """
        Runs langdetect on a sample of the sentences.
        Returns the most commonly detected language.
        """
        sentences = [s for s in self.sentences_clean if len(s)>10]
        sample_len = 20
        if len(sentences)<=sample_len:
            sentences_sample = sentences
        else:
            rand_start = randint(0, (len(sentences)-1)-sample_len)
            sentences_sample = sentences[rand_start:rand_start+sample_len]
        detections = []
        for i in sentences_sample:
            try:
                detections.append(detect(i))
            except LangDetectException:
                pass
        if len(detections)==0:
            raise ValueError('Insufficient data to detect language!')
        lang = cnt(detections).most_common(1)[0][0]

        if lang not in self.supported_languages:
            raise ValueError('Lang not supported!')
        return lang
    
    def get_pos_tokens(self):
        """
        Get part-of-speech tags of each token in sentence
        """
        # Get spaCy model
        spacy_model = get_spacy_model(self.lang)
        # Pipe sentences
        sdocs = list(spacy_model.pipe(self.sentences_clean))
        # Concatenate Docs
        sc_doc = Doc.from_docs(sdocs)
        # Extract text and pos from Doc
        pos_tokens = [(token.text, token.pos_) for token in sc_doc]
        return pos_tokens
    
    def get_pos_ngrams(self, ngrams_min=1, ngrams_max=2):
        """
        Get ngrams from part-of-speech tagged sentences
        """
        pos_tokens = self.get_pos_tokens()
        pos_ngrams = (zip(*[pos_tokens[i:] for i in range(n)])
                  for n in range(ngrams_min, ngrams_max+1))
        return (ng for ngl in pos_ngrams for ng in ngl)
    
    def get_candidate_ngrams(self,
                             min_freq=1,
                             include_pos=None,
                             exclude_pos=None,
                             **kwargs):
        """
        Get final candidate ngrams from part-of-speech tagged sentences
        """
        pos_ngrams = self.get_pos_ngrams(**kwargs)
        pos_ngrams = [a for a,b in cnt(list(pos_ngrams)).items() if b>=min_freq]
        candidate_ngrams = filter_ngrams(pos_ngrams, include_pos=include_pos)
        return candidate_ngrams
    
    def get_embeddings(self):
        embs = model.encode(self.candidate_ngrams['joined_ngrams'])
        return embs
    

class BiText:
    """
    Implements fast methods for bilingual terminology extraction.
    Bitext is two Texts.
    """
    def __init__(self, bitext):
        self.src_text = Text(bitext['source'].tolist())
        self.trg_text = Text(bitext['target'].tolist())
        self.src_ngrams_df = self.src_text.get_candidate_ngrams()
        self.trg_ngrams_df = self.trg_text.get_candidate_ngrams()
        self.src_embs = self.src_text.get_embeddings()
        self.trg_embs = self.trg_text.get_embeddings()
    
    @staticmethod
    def get_seq_similarities(src_embeddings, trg_embeddings):
        # Get source/target ngram similarity matrix
        src_seq_similarities = cosine_similarity(src_embeddings,
                                              trg_embeddings)
        
        # Get target/source ngram similarity matrix
        trg_seq_similarities = cosine_similarity(trg_embeddings,
                                              src_embeddings)
        return src_seq_similarities, trg_seq_similarities

    def get_aligned_ngrams(self, min_similarity=.8, **kwargs):
        """
        Get top bitext ngrams from one side
        """
        src_ngrams = self.src_ngrams_df['joined_ngrams'].tolist()
        trg_ngrams = self.trg_ngrams_df['joined_ngrams'].tolist()
        # Get POS tags
        src_tags = self.src_ngrams_df['tags'].tolist()
        trg_tags = self.trg_ngrams_df['tags'].tolist()
        # Get embeddings
        src_embeddings = self.src_embs#model.encode(src_ngrams)
        trg_embeddings = self.trg_embs#odel.encode(trg_ngrams)
        
        # Get similarities
        src_seq_similarities, trg_seq_similarities = self.get_seq_similarities(src_embeddings,
                                                                               trg_embeddings)
        # Get indexes
        src_idx = list(range(len(src_ngrams)))
        trg_idx = list(range(len(trg_ngrams)))
   
        # Get indexes and values of most similar source ngram for each target ngram
        src_max_values = np.max(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        src_max_idx = np.argmax(trg_seq_similarities[trg_idx][:, src_idx], axis=1)
        
        # Get indexes and values of most similar target ngram for each source ngram
        trg_max_values = np.max(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        trg_max_idx = np.argmax(src_seq_similarities[src_idx][:, trg_idx], axis=1)
        
        # make ngrams dataframe with the top src_ngram/trg_ngram similarities
        src_aligned_ngrams = pd.DataFrame([(src_ngrams[idx],
                                            src_tags[idx],
                                            trg_ngrams[trg_max_idx[idx]],
                                            trg_tags[trg_max_idx[idx]],
                                            float(trg_max_values[idx])) for idx in src_idx])
        
        # make ngrams dataframe with the top trg_ngram/src_ngram similarities
        trg_aligned_ngrams = pd.DataFrame([(src_ngrams[src_max_idx[idx]],
                                            src_tags[src_max_idx[idx]],
                                            trg_ngrams[idx],
                                            trg_tags[idx],
                                            float(src_max_values[idx])) for idx in trg_idx])
        
        return src_aligned_ngrams, trg_aligned_ngrams

    def get_top_ngrams(self, min_similarity=.8, **kwargs):
        """
        Extract and filter all source ngrams and all target ngrams.
        Find their most similar matches.
        Much faster, less precise, can cause OOM errors.
        """
        src_aligned_ngrams, trg_aligned_ngrams = self.get_aligned_ngrams(**kwargs)
        
        bi_ngrams = pd.concat([src_aligned_ngrams, trg_aligned_ngrams])
        bi_ngrams = bi_ngrams.reset_index()
        bi_ngrams = bi_ngrams.drop(columns=['index'])
        bi_ngrams.columns = ['src_ngram',
                             'src_ngram_tags',
                             'trg_ngram',
                             'trg_ngram_tags',
                             'bi_ngram_similarity']
        
        # Keep n-grams above min_similarity
        bi_ngrams = bi_ngrams[bi_ngrams['bi_ngram_similarity'] >= min_similarity]
        if len(bi_ngrams)==0:
            raise ValueError('No ngram pairs above minimum similarity!')
        
        # For one-word terms, keep those longer than 1 character
        bi_ngrams = bi_ngrams[bi_ngrams['src_ngram'].str.len()>1]
        bi_ngrams = bi_ngrams[bi_ngrams['trg_ngram'].str.len()>1]
        
        # Group by source, get the most similar target n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (src_ngram, df) in list(bi_ngrams.groupby('src_ngram'))])
        
        # Group by target, get the most similar source n-gram
        bi_ngrams = pd.DataFrame([df.loc[df['bi_ngram_similarity'].idxmax()]
                            for (trg_ngram, df) in list(bi_ngrams.groupby('trg_ngram'))])
        
        bi_ngrams = bi_ngrams.round(4)
     
        return bi_ngrams

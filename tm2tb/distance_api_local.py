"""
Similarity API. Based on sim_server_remote.py
Takes two sequences of strings.
Returns the two sequences and their distances.
"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
MODEL_PATH = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_server/labse_model'
model = SentenceTransformer(MODEL_PATH)

class DistanceApi:
    """
    Takes two sequences of strings.
    Extract distances.
    Calculate Maximal Marginal Relevance (MMR).
    Return closest sequences.
    
    seq1: ['dog', 'cat', 'horse']
    seq2: ['gato', 'caballo', 'perro']

    result = [(dog, perro, .1), (cat, gato, .09), (horse, caballo, .11)]
    """
    def __init__(self, request):
        self.input_data = json.loads(request)
        self.seq1 = self.input_data['seq1']
        self.seq2 = self.input_data['seq2']
        self.diversity = self.input_data['diversity']
        self.top_n = self.input_data['top_n']
        self.seq1_embeddings = model.encode(self.seq1)
        self.seq2_embeddings = model.encode(self.seq2)

    def get_closest_sequence_elements(self):
        'Return distances and index of seq1 and seq2 embeddings'
        seq1_embeddings = self.seq1_embeddings
        seq2_embeddings = self.seq2_embeddings
        index = faiss.IndexFlatL2(seq2_embeddings.shape[1])
        index.add(seq1_embeddings)
        D, I = index.search(seq2_embeddings, 1)
        ordered_seq2 = [self.seq1[i[0]] for i in I]
        result = list(zip(self.seq2, ordered_seq2, D))
        return [(a, b, float(c)) for (a, b, c) in result]

    def get_ngram_sentence_dists(self):
        sentence_embedding = self.seq1_embeddings
        ngram_embeddings = self.seq2_embeddings
        index = faiss.IndexFlatL2(ngram_embeddings.shape[1])
        index.add(sentence_embedding)
        D, I = index.search(ngram_embeddings, 1)
        return D, I
    
    def get_ngram_dists(self):
        ngram_embeddings = self.seq2_embeddings
        index = faiss.IndexFlatL2(ngram_embeddings.shape[1])
        index.add(ngram_embeddings)
        D, I = index.search(ngram_embeddings, len(ngram_embeddings))
        return D, I

    def get_top_sentence_ngrams(self):
        # Based on KeyBERT: https://github.com/MaartenGr/KeyBERT/blob/master/keybert/_mmr.py
        # Extract distances within ngrams, and between ngrams and the sentence
        ngram_sentence_dists, _ = self.get_ngram_sentence_dists()
        ngram_dists, _ = self.get_ngram_dists()
        top_n = self.top_n
        ngrams = self.seq2
        diversity = self.diversity
        # Initialize candidates and choose best ngram
        best_ngrams_idx = [np.argmin(ngram_sentence_dists)]
        
        # All keywords that are not in best keywords
        candidates_idx = [i for i in range(len(ngrams)) if i != best_ngrams_idx[0]]
        
        for _ in range(min(top_n - 1, len(ngrams) - 1)):
            # Get distances within candidates and between candidates and selected ngrams
            candidate_similarities = ngram_sentence_dists[candidates_idx, :]
            target_similarities = np.min(ngram_dists[candidates_idx][:, best_ngrams_idx], axis=1)
            
            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            
            # Get closest candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]
            
            # Update best ngrams & candidates
            best_ngrams_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        
        # Get sorted best ngrams
        best_ngrams = sorted([(ngrams[idx], round(float(ngram_sentence_dists.reshape(1, -1)[0][idx]), 4)) 
                       for idx in best_ngrams_idx], key=lambda tup: tup[1])
        return best_ngrams
    








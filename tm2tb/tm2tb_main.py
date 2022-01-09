"""
Similarity API. Based on sim_server_remote.py
Takes two sequences of strings.
Returns the two sequences and their distances.
"""
import json
#import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tm2tb import Sentence
from tm2tb import BiSentence
from tm2tb import BiText
from tm2tb import BilingualReader
from sentence_transformers import SentenceTransformer
MODEL_PATH = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_server/labse_model'
model = SentenceTransformer(MODEL_PATH)

#%%
# example sentence test

class Tm2Tb:    
    """
    main class to do all stuff
    """
    def __init__(self, model=model):
        self.model = model
        
    def get_sentence_ngrams(self, sentence, **kwargs):
        sn = Sentence(sentence)
        ngrams = sn.get_ngrams(**kwargs)
        return ngrams
    
    def get_top_sentence_ngrams_remote(self, sentence, **kwargs):
        sn = Sentence(sentence)
        ngrams = sn.get_top_ngrams(**kwargs)
        return ngrams
    
    def get_top_sentence_ngrams(self, sentence, diversity=.5, top_n = 50, **kwargs):
        sn = Sentence(sentence)
        ngrams = sn.get_ngrams(**kwargs)
        sentence = sn.clean_sentence
        ngrams = list(set(ngrams['joined_ngrams']))

        seq1_embeddings = model.encode([sentence])
        seq2_embeddings = model.encode(ngrams)
        
        ngram_sentence_dists = cosine_similarity(seq2_embeddings, seq1_embeddings)
        ngram_dists = cosine_similarity(seq2_embeddings)
        
        # Initialize candidates and choose best ngram
        best_ngrams_idx = [np.argmax(ngram_sentence_dists)]
        
        # All keywords that are not in best keywords
        candidates_idx = [i for i in range(len(ngrams)) if i != best_ngrams_idx[0]]
        
        for _ in range(min(top_n - 1, len(ngrams) - 1)):
            # Get distances within candidates and between candidates and selected ngrams
            candidate_similarities = ngram_sentence_dists[candidates_idx, :]
            
            target_similarities = np.max(ngram_dists[candidates_idx][:, best_ngrams_idx], axis=1)
            
            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            
            # Get closest candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]
            
            # Update best ngrams & candidates
            best_ngrams_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        
        # Get sorted best ngrams
        top_sentence_ngrams = sorted([(ngrams[idx], round(float(ngram_sentence_dists.reshape(1, -1)[0][idx]), 4)) 
                       for idx in best_ngrams_idx], key=lambda tup: tup[1], reverse=True)

        return top_sentence_ngrams


    def get_bilingual_ngrams(self, src_sentence, trg_sentence, min_similarity=.75, **kwargs):
        
        src_ngrams, _ = zip(*self.get_top_sentence_ngrams(src_sentence, **kwargs))
        trg_ngrams, _ = zip(*self.get_top_sentence_ngrams(trg_sentence, **kwargs))
        
        seq1_embeddings = model.encode(src_ngrams)#return embs
        seq2_embeddings = model.encode(trg_ngrams)
        
        seq_similarities = cosine_similarity(seq1_embeddings,
                                             seq2_embeddings)
        
        # get seq1 & seq2 indexes
        seq1_idx = [i for i in range(len(src_ngrams))]
        seq2_idx = [i for i in range(len(trg_ngrams))]
        
        # get max seq2 values and indexes
        max_seq2_values = np.max(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
        max_seq2_idx = np.argmax(seq_similarities[seq1_idx][:, seq2_idx], axis=1)
        
        # get max seq similarities
        max_seq_similarities = [(src_ngrams[idx], trg_ngrams[max_seq2_idx[idx]], 
                                  float(round(max_seq2_values[idx], 4))) for idx in seq1_idx]
        
        # sort max seq similarities
        max_seq_similarities = sorted(max_seq_similarities, key=lambda tup: tup[2], reverse=True)
        
        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(max_seq_similarities)
        bilingual_ngrams.columns = ['src', 'trg', 'similarity']

        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        # Filter by distance
        bilingual_ngrams = bilingual_ngrams[bilingual_ngrams['similarity'] >= min_similarity]
        bilingual_ngrams = bilingual_ngrams.sort_values(by='similarity', ascending=False)

        return bilingual_ngrams
    
    def get_bilingual_ngrams_remote(self, src_sentence, trg_sentence, **kwargs):
        bisentence = BiSentence(src_sentence, trg_sentence)
        bilingual_ngrams = (bisentence.get_bilingual_ngrams(server_mode='remote'))
        return bilingual_ngrams
    
    def get_bitext_bilingual_ngrams_precise(self,
                                            path,
                                            file_name,
                                         diversity=.5,
                                         top_n=8,
                                         min_similarity=.8,
                                         **kwargs):
        
        bitext = BilingualReader(path, file_name).get_bitext()
        
        all_bilingual_ngrams = []
        for i in range(len(bitext)):
            try:
                src_row = bitext.iloc[i]['src']
                trg_row = bitext.iloc[i]['trg']
                bilingual_ngrams = self.get_bilingual_ngrams(src_row, trg_row)
                bilingual_ngrams = pd.DataFrame(bilingual_ngrams)
                bilingual_ngrams.columns = ['src','trg','similarity']
                all_bilingual_ngrams.append(bilingual_ngrams)
            except:
                pass
        if len(all_bilingual_ngrams)==0:
            raise ValueError("No bitext_bilingual_ngrams from get_bitext_bilingual_ngrams_precise")
          
        bitext_bilingual_ngrams = pd.concat(all_bilingual_ngrams)
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.drop_duplicates(subset='src')
        bitext_bilingual_ngrams = bitext_bilingual_ngrams.reset_index()
        bitext_bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (src_ngram, df) in list(bitext_bilingual_ngrams.groupby('src'))])

        # Group by target ngram, get most similar source ngram
        bitext_bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmin()]
                            for (trg_ngram, df) in list(bitext_bilingual_ngrams.groupby('trg'))])

        # Filter by similarity
        bitext_bilingual_ngrams = bitext_bilingual_ngrams[bitext_bilingual_ngrams['similarity'] >= min_similarity]

        return bitext_bilingual_ngrams
    
    def get_bitext_bilingual_ngrams_precise_remote(self,
                                            path,
                                            file_name,
                                         diversity=.5,
                                         top_n=8,
                                         min_similarity=.8,
                                         **kwargs):
        bt = BiText(path, file_name)
        bitermsp = bt.get_bitext_bilingual_ngrams_precise(server_mode="remote")
        return bitermsp

tt = Tm2Tb()

#%%

print('TM2TB can extract bilingual terms from documents such as translation memories:')

path = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_webapp/uploads'
file_name = '22016A1019.tmx_en-es.csv'

btbng = tt.get_bitext_bilingual_ngrams_precise(path, file_name)
print(btbng)

#%%
print('I dont have a bilingual document, I just want to get terms from a pair of sentences.')
print('Sure. You can also extract bilingual terms from pairs of translated sentences:')

src_sentence = """ The giant panda, also known as the panda bear (or simply the panda), is a bear native to South Central China. It is characterised by its bold black-and-white coat and rotund body. The name "giant panda" is sometimes used to distinguish it from the red panda, a neighboring musteloid. Though it belongs to the order Carnivora, the giant panda is a folivore, with bamboo shoots and leaves making up more than 99% of its diet. Giant pandas in the wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, rodents, or carrion. In captivity, they may receive honey, eggs, fish, shrub leaves, oranges, or bananas. """

trg_sentence = """
El panda gigante, también conocido como oso panda (o simplemente panda), es un oso originario del centro-sur de China. Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre de "panda gigante" se usa en ocasiones para distinguirlo del panda rojo, un mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda gigante es folívoro, y más del 99 % de su dieta consiste en brotes y hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas de arbustos, naranjas o plátanos."""

bng = tt.get_bilingual_ngrams(src_sentence, trg_sentence)
print(bng)

#%%
# print('Some results are wrong. Why?')
# print('You can set the minimum similarity value for the results. The highest the similarity, the higher the probability that the translation is correct. Lets get terms with a minimum similarity of .9:')

# bng = tt.get_bilingual_ngrams(src_sentence, trg_sentence, min_similarity=.9)
# print(bng)


#%% REMOTE

print('TM2TB can extract bilingual terms from documents such as translation memories:')

path = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_webapp/uploads'
file_name = '22016A1019.tmx_en-es.csv'

btbng = tt.get_bitext_bilingual_ngrams_precise_remote(path, file_name)
print(btbng)

# bng = tt.get_bilingual_ngrams_remote(src_sentence, trg_sentence)
# print(bng)

print('I just want to get terms/keywords from one sentence.')
print('Sure. You can also use TM2TB as a monolingual keyword extractor:')
src_tsng = tt.get_top_sentence_ngrams_remote(src_sentence)
print(src_tsng)

trg_tsng = tt.get_top_sentence_ngrams_remote(trg_sentence)
print(trg_tsng)

#%%
print('Extracting from a big document is too slow. Can it get faster?')
print('Yes, pass "fast" to get a faster term extraction')

#%%
print('I would like to get longer terms, not only single words. How can I get them?')
print('You can get terms of length 2, 3, or more with ngrams_min and ngrams_max:')

bng = tt.get_bilingual_ngrams(src_sentence, trg_sentence, ngrams_min=2, ngrams_max=3)
print(bng)

#%%
print('Does TM2TB know about grammatical categories of words? Can I get terms that are nouns or adjectives, for example?')

print('Yes. You can pass a list of Part-of-Speech tags to delimit the ngrams:')
print('Lets get only nouns and proper nouns:')
bng = tt.get_bilingual_ngrams(src_sentence, trg_sentence, include_pos=['NOUN','PROPN'])
print(bng)

print('Lets get only adjectives and adverbs:')
bng = tt.get_bilingual_ngrams(src_sentence, trg_sentence, include_pos=['ADJ'])
print(bng)

#%%
print('I just want to get terms/keywords from one sentence.')
print('You can also use TM2TB as a monolingual keyword extractor:')
src_tsng = tt.get_top_sentence_ngrams(src_sentence)
print(src_tsng)

trg_tsng = tt.get_top_sentence_ngrams(trg_sentence)
print(trg_tsng)
#%% FAQ
print('What do the numbers mean?')
print('The numbers represent a measure known as cosine similarity.')

print('How does is work?')
print('TM2TB extracts ngrams from the sentences, embeds them using a transformer model and compares them to get the most similar terms.')


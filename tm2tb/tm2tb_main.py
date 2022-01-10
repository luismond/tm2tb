"""
"""
import json
#import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tm2tb import Sentence
# from tm2tb import BiSentence
# from tm2tb import BiText
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

    def get_sentence_ngrams(self,
                                sentence,                                
                                ngrams_min=1,
                                ngrams_max=3,
                                include_pos=None,
                                exclude_pos=None,
                                top_n = 25,
                                diversity=.5):
        """
        Get the ngrams that are most similar to the sentence.

        Parameters
        ----------
        server_mode : string, optional
            DESCRIPTION. Defines if the similarity queries are done locally or remotely.
        diversity : int, optional
            DESCRIPTION. Diversity value for Maximal Marginal Relevance. The default is .8.
        top_n : int, optional
            DESCRIPTION. Number of best ngrams to return. The default is 20.
        overlap : string, optional
            DESCRIPTION. Defines if overlapping ngrams should be kept or dropped.
        **kwargs : dict
            DESCRIPTION. Same optional parameters as Sentence.get_ngrams()
        ngrams_min : int, optional
            DESCRIPTION. Minimum ngram sequence length.
        ngrams_max : int, optional
            DESCRIPTION. Maximum ngram sequence length.
        include_pos : List, optional
            DESCRIPTION. A list of POS-tags to delimit the ngrams.
                        If None, the default value is ['NOUN', 'PROPN']
        exclude_pos : List, optional
            DESCRIPTION. A list of POS-tags to exclude from the ngrams.
                        If None, the default value is ['X', 'SCONJ', 'CCONJ', 'AUX']
        Returns
        -------
        top_ngrams : List of tuples (ngram, value).
            List of top_n ngrams that are most similar to the sentence.
        """
        sn = Sentence(sentence)
        ngrams = sn.get_ngrams(ngrams_min=ngrams_min,
                               ngrams_max=ngrams_max,
                               include_pos=include_pos,
                               exclude_pos=exclude_pos)
        
        sentence = sn.clean_sentence
        ngrams = list(set(ngrams['joined_ngrams']))

        seq1_embeddings = model.encode([sentence])
        seq2_embeddings = model.encode(ngrams)
        
        ngram_sentence_dists = cosine_similarity(seq2_embeddings, seq1_embeddings)
        ngram_dists = cosine_similarity(seq2_embeddings)
        
        # Initialize candidates and choose best ngram
        best_ngrams_idx = [np.argmax(ngram_sentence_dists)]
        
        # All ngrams that are not in best ngrams
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
        
        # Get best ngrams
        top_sentence_ngrams = [(ngrams[idx], 
                                round(float(ngram_sentence_dists.reshape(1, -1)[0][idx]), 4))
                               for idx in best_ngrams_idx]

        return sorted(top_sentence_ngrams, key=lambda tup: tup[1], reverse=True)


    def get_bilingual_ngrams(self,
                             src_sentence,
                             trg_sentence,
                             min_similarity=.75,
                             **kwargs):
        
        
        
        src_ngrams, sv = zip(*self.get_sentence_ngrams(src_sentence, **kwargs))
        trg_ngrams, tv = zip(*self.get_sentence_ngrams(trg_sentence, **kwargs))
        
        sd = dict(zip(src_ngrams,sv))
        td = dict(zip(trg_ngrams,sv))

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
        
        bilingual_ngrams['src_s'] = bilingual_ngrams['src'].apply(lambda x: sd[x])
        bilingual_ngrams['trg_s'] = bilingual_ngrams['trg'].apply(lambda x: td[x])
        
        # Turn to list
        bilingual_ngrams = list(zip(bilingual_ngrams['src'],bilingual_ngrams['trg'],
                                    bilingual_ngrams['similarity'],bilingual_ngrams['src_s'],
                                    bilingual_ngrams['trg_s']))
 
        bilingual_ngrams = [(a, b, round(c, 4), d, e) for (a,b,c,d,e) in bilingual_ngrams]
        

        return bilingual_ngrams

    
    def get_bilingual_ngrams_from_bitext(self,
                                         path,
                                         file_name,
                                         **kwargs):
        
        bitext = BilingualReader(path, file_name).get_bitext()
        print(bitext[:10])
        all_bilingual_ngrams = []
        for i in range(len(bitext)):
            try:
                src_row = bitext.iloc[i]['src']
                trg_row = bitext.iloc[i]['trg']
                bilingual_ngrams = self.get_bilingual_ngrams(src_row, trg_row)
                for bn in bilingual_ngrams:
                    if not bn in all_bilingual_ngrams:
                        all_bilingual_ngrams.append(bn)
            except:
                pass

        # Make bilingual_ngrams dataframe
        bilingual_ngrams = pd.DataFrame(all_bilingual_ngrams)
        bilingual_ngrams.columns = ['src', 'trg', 'similarity', 'src_s','trg_s']
        
        # Group by source, get closest target ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (src_ngram, df) in list(bilingual_ngrams.groupby('src'))])

        # Group by target, get closest source ngram
        bilingual_ngrams = pd.DataFrame([df.loc[df['similarity'].idxmax()]
                            for (trg_ngram, df) in list(bilingual_ngrams.groupby('trg'))])

        bilingual_ngrams['x'] = bilingual_ngrams['similarity']*bilingual_ngrams['src_s']*bilingual_ngrams['trg_s']
        # Filter by distance
        bilingual_ngrams = bilingual_ngrams.sort_values(by='x', ascending=False)
        
        #Turn to list
        bilingual_ngrams = list(zip(bilingual_ngrams['src'],bilingual_ngrams['trg'],bilingual_ngrams['similarity'],
                                    bilingual_ngrams['src_s'],bilingual_ngrams['trg_s']))
            
        return bilingual_ngrams
    
    def get_bilingual_ngrams_from_bitext_fast(self,
                                         path,
                                         file_name,
                                         **kwargs):
        
        bitext = BilingualReader(path, file_name).get_bitext()
        # get all src & trg ngrams
        def get_src_trg_top_ngrams():
            bitext_src_top_ngrams = []
            bitext_trg_top_ngrams = []
            for i in range(len(bitext)):
                try:
                    ss = bitext.iloc[i]['src']
                    src_top_ngrams = self.get_sentence_ngrams(ss, top_n=30)
    
                    for ngram in src_top_ngrams:
                        if not ngram in bitext_src_top_ngrams:
                            bitext_src_top_ngrams.append(ngram)
    
                    ts = bitext.iloc[i]['trg']
                    trg_top_ngrams  = self.get_sentence_ngrams(ts, top_n=30)
                    
                    for ngram in trg_top_ngrams:
                        if not ngram in bitext_trg_top_ngrams:
                            bitext_trg_top_ngrams.append(ngram)
                except:
                    pass
            # if len(bitext_src_top_ngrams) == 0 or len(bitext_trg_top_ngrams)==0:
            #     raise ValueError('No bitext_bilingual_ngrams from get_bitext_bilingual_ngrams_fast')
            return bitext_src_top_ngrams, bitext_trg_top_ngrams
    
       
        
        src_ngramsx, trg_ngramsx = get_src_trg_top_ngrams()
        src_ngrams = [a for (a,b) in src_ngramsx]
        trg_ngrams = [a for (a,b) in trg_ngramsx]
        
        src_values = [b for (a,b) in src_ngramsx]
        trg_values = [b for (a,b) in trg_ngramsx]
       
        sd = dict(zip(src_ngrams,src_values))
        td = dict(zip(trg_ngrams,trg_values))
        
        print(src_ngrams)
        print(trg_ngrams)
        
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

       
        bilingual_ngrams['src_s'] = bilingual_ngrams['src'].apply(lambda x: sd[x])
        bilingual_ngrams['trg_s'] = bilingual_ngrams['trg'].apply(lambda x: td[x])
        
        bilingual_ngrams['x'] = bilingual_ngrams['similarity']*bilingual_ngrams['src_s']*bilingual_ngrams['trg_s']
        # Filter by distance
        bilingual_ngrams = bilingual_ngrams.sort_values(by='x', ascending=False)
        
        
        # # Turn to list
        bilingual_ngrams = list(zip(bilingual_ngrams['src'],bilingual_ngrams['trg'],
                                    bilingual_ngrams['similarity'],bilingual_ngrams['src_s'],
                                    bilingual_ngrams['trg_s']))
 
        bilingual_ngrams = [(a, b, round(c, 4), d, e) for (a,b,c,d,e) in bilingual_ngrams]
        
        
        return bilingual_ngrams
    
    def get_top_sentence_ngrams_remote(self, sentence, **kwargs):
        sn = Sentence(sentence)
        ngrams = sn.get_top_ngrams(**kwargs)
        return ngrams
    
    def get_bilingual_ngrams_remote(self, src_sentence, trg_sentence, **kwargs):
        bisentence = BiSentence(src_sentence, trg_sentence)
        bilingual_ngrams = (bisentence.get_bilingual_ngrams(server_mode='remote'))
        return bilingual_ngrams
    
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



#%% SENTENCE
from pprint import pp
print('TM2TB is a term/keyword/phrase extractor. It can extract terms from a sentence:\n')

sentence = """The giant panda, also known as the panda bear (or simply the panda), is a bear native to South Central China. It is characterised by its bold black-and-white coat and rotund body. The name "giant panda" is sometimes used to distinguish it from the red panda, a neighboring musteloid. Though it belongs to the order Carnivora, the giant panda is a folivore, with bamboo shoots and leaves making up more than 99% of its diet. Giant pandas in the wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, rodents, or carrion. In captivity, they may receive honey, eggs, fish, shrub leaves, oranges, or bananas.\n"""

print(sentence)


#print('Without arguments:\n')
src_ng = tt.get_sentence_ngrams(sentence)
pp(src_ng[:15])

#print('\nWith arguments:\n')
src_ng = tt.get_sentence_ngrams(sentence,
                                    ngrams_min=1,
                                    ngrams_max=3,
                                    include_pos=['NOUN','ADJ'],
                                    top_n =15,
                                    diversity=.9)

#pp(src_ng)
print('\nThe values represent the similarity between the term and the sentence.')


#%% BISENTENCE

print('TM2TB supports many languages.')
print('Lets get terms from the translation to Spanish of the above sentence:\n')

translated_sentence = """El panda gigante, también conocido como oso panda (o simplemente panda), es un oso originario del centro-sur de China. Se caracteriza por su llamativo pelaje blanco y negro, y su cuerpo rotundo. El nombre de "panda gigante" se usa en ocasiones para distinguirlo del panda rojo, un mustélido parecido. Aunque pertenece al orden de los carnívoros, el panda gigante es folívoro, y más del 99 % de su dieta consiste en brotes y hojas de bambú. En la naturaleza, los pandas gigantes comen ocasionalmente otras hierbas, tubérculos silvestres o incluso carne de aves, roedores o carroña. En cautividad, pueden alimentarse de miel, huevos, pescado, hojas de arbustos, naranjas o plátanos.\n"""
print(translated_sentence)
trg_ng = tt.get_sentence_ngrams(translated_sentence)
pp(trg_ng[:15])

#
print('But the special thing about TM2TB is that it can match the terms from a sentence and a translated sentence:\n')

bng = tt.get_bilingual_ngrams(sentence, translated_sentence, top_n=50)
pp(bng)

#%% BITEXT

print('Furthermore, TM2TB can also extract bilingual terms from bilingual documents. Lets take a small translation file:\n')
path = '/home/user/pCloudDrive/PROGRAMMING/APPS/TM2TB/tm2tb_client/tests/data/PandaExamples'
file_name = 'panda.txt_spa-MX.mqxliff'
btbng = tt.get_bilingual_ngrams_from_bitext(path, file_name)
pp(btbng[:20])

#%%
btbng = tt.get_bilingual_ngrams_from_bitext_fast(path, file_name)


#%%
#print('Extracting from a big document is too slow. Can it get faster?')
#print('Yes, pass "fast" to get a faster term extraction')

print('I would like to get longer terms, not only single words. How can I get them?\n')
print('You can get terms of length 2, 3, or more with ngrams_min and ngrams_max:\n')

bng = tt.get_bilingual_ngrams(sentence, translated_sentence, ngrams_min=2, ngrams_max=3)
pp(bng)


#%%
print('Does TM2TB know about grammatical categories of words? Can I get terms that are nouns or adjectives, for example?\n')

print('Yes. You can pass a list of Part-of-Speech tags to delimit the ngrams:\n')
print('Lets get only nouns and proper nouns:')
bng = tt.get_bilingual_ngrams(sentence, translated_sentence, include_pos=['NOUN','PROPN'])
pp(bng)

print('Lets get only adjectives and adverbs:\n')
bng = tt.get_bilingual_ngrams(sentence, translated_sentence, include_pos=['ADJ'])
pp(bng)

#%% FAQ

print('I dont have a bilingual document, I just want to get terms from a pair of sentences.')
print('Sure. You can also extract bilingual terms from pairs of translated sentences:')


print('What do the numbers mean?')
print('The numbers represent a measure known as cosine similarity.')

print('How does is work?')
print('TM2TB extracts ngrams from the sentences, embeds them using a transformer model and compares them to get the most similar terms.')


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

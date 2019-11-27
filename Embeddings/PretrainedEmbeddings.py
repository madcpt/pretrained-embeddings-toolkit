import torch
from torch import nn
from embeddings import FastTextEmbedding, GloveEmbedding, KazumaCharEmbedding
 
class PretrainedEmbeddings(object):
    def __init__(self):
        self.embedding_models = []
        self.supported_models = ['FastText', 'Glove', 'KazumaChar', 'SL999']
        print('Welcome, supported models: ', end='')
        print(self.supported_models)
    
    def add_pretrained_models(self, models: []):
        self.embedding_models = []
        for model in models:
            if model == self.supported_models[0]:
                self.embedding_models.append(FastTextEmbedding())
            if model == self.supported_models[1]:
                self.embedding_models.append(GloveEmbedding())
            if model == self.supported_models[2]:
                self.embedding_models.append(KazumaCharEmbedding())
            if model == self.supported_models[3]:
                self.embedding_models.append(SL999())
            pass
        
    def dump_pretrained_emb(self, word2index, index2word, dump_path):
        print("Dumping pretrained embeddings...")
        E = []
        for i in tqdm(range(len(word2index.keys()))):
            w = index2word[i]
            e = []
            for emb in self.embeddings:
                e += emb.emb(w, default='zero')
            E.append(e)
        with open(dump_path, 'wt') as f:
            json.dump(E, f)
import sys
import json
from tqdm import tqdm
from embeddings import KazumaCharEmbedding
import yaml

UNK = 'UUUNKKK'

def read_file():
    with open('./data/paragram_300_sl999.txt', 'r', encoding = "ISO-8859-1") as f:
        lines = f.readlines()
    wordlist = []
    params = []
    for line in tqdm(lines, total=len(lines)):
        line_params = line.split(' ')
        wordlist.append(line_params[0])
        params.append([float(param) for param in line_params[1:]])
    return wordlist, params


def dump_params(wordlist, params):
    print("dumping embedding")
    with open('./data/sl999_wordlist.json', 'w') as f:
        json.dump(wordlist, f)
    with open('./data/sl999.emb', 'w') as f:
        json.dump(params, f)

def read_params():
    print("loading vocab")
    with open('./data/sl999_wordlist.json', 'r') as f:
        wordlist = json.load(f)
    print("loading pretrained parameters")
    with open('./data/sl999.emb', 'r') as f:
        params = json.load(f)
    index2word = {}
    word2index = {}
    for word in tqdm(wordlist, total=len(wordlist)):
        word2index[word] = len(word2index)
        index2word[len(index2word)] = word

    print("loading vocab word2index")
    with open('./data/word2index.json', 'r') as f:
        voc_word2index = json.load(f)
    
    return word2index, index2word, params, voc_word2index

def get_embeddings(voc_word2index, params, word2index):
    k = KazumaCharEmbedding()
    embed_400d = []
    for voc_word in tqdm(voc_word2index.keys(), total=len(voc_word2index)):
        if voc_word in word2index.keys():
            wordid = word2index[voc_word]
        else:
            wordid = word2index[UNK]
        sl_emb = params[wordid]
        k_emb = k.emb(voc_word)
        cat_emb = sl_emb + k_emb
        embed_400d.append(cat_emb)
        if voc_word == 'PAD':
            print(embed_400d)
    print("dumping")
    with open('./data/embed_{}.json'.format(str(len(voc_word2index))), 'w') as f:
        json.dump(embed_400d, f)
        

if __name__ == "__main__":
    # wordlist, params = read_file()
    # dump_params(wordlist, params)
    # word2index, index2word, params, voc_word2index = read_params()
    # get_embeddings(voc_word2index, params, word2index)

    
    with open('./config/sl999-config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        fruits_list = yaml.load(file, Loader=yaml.FullLoader)

        print(fruits_list)
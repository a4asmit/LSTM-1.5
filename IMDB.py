import torch
import torch.nn as nn
import torch.optim as otpim
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE=10000 
EMBED_DIM=128 
HIDDEN_SIZE=256

NUM_LAYERS = 2
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenize(text):
    return re.sub(r"[^a-zA-Z ]", "", text.lower()).split()

def build_vocab(texts, max_vocab):
    counter = Counter(t for text in texts for t in tokenize(text))
    vocab = {'<PAD>' : 0, '<UNK>' : 1}

    for word, _ in counter.most_common(max_vocab-2):
        vocab[word] = len(vocab)

    return vocab 

def encode(text, vocab, max_len):
    ids = [vocab.get(t,1) for t in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids)) 

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [(torch.tensor(encode(t, vocab, max_len)), l for t, l in zip(texts, labels))]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    


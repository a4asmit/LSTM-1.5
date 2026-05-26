import torch
import torch.nn as nn
import torch.optim as optim
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
        self.data = [(torch.tensor(encode(t, vocab, max_len)), l) for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    

class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=0.3, bidirectional=True)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)

        return self.fc(self.dropout(hidden)).squeeze(1)
        

model = SentimentLSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += ((pred > 0).float() == y).sum().item()
    return total_loss/len(loader), correct/len(loader.dataset)
torch.save(model.state_dict(), 'sentiment_lstm.pth')
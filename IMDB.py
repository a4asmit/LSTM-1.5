import torch
import torch.nn as nn
import torch.optim as optim
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ============================================
# CONFIG
# ============================================

VOCAB_SIZE = 10000 
EMBED_DIM = 128 
HIDDEN_SIZE = 128
NUM_LAYERS = 1
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================
# TOKENIZER
# ============================================

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

# ============================================
# DATASET
# ============================================

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [(torch.tensor(encode(t, vocab, max_len)), l) 
                     for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

# ============================================
# MODEL
# ============================================

class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(self.dropout(hidden)).squeeze(1)

# ============================================
# DUMMY DATA
# ============================================

def create_dummy_data(num_samples=1000):
    positive = [
        "this movie was absolutely amazing",
        "i loved this film it was great",
        "best movie ever highly recommend"
    ]
    negative = [
        "this movie was terrible and awful",
        "worst movie ever made very bad",
        "boring waste of time"
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:
            texts.append(positive[i % len(positive)] + f" {i}")
            labels.append(1)
        else:
            texts.append(negative[i % len(negative)] + f" {i}")
            labels.append(0)
    
    return texts, labels

# ============================================
# TRAINING
# ============================================

print("Creating data...")
all_texts, all_labels = create_dummy_data(num_samples=2000)

print("Building vocab...")
vocab = build_vocab(all_texts, VOCAB_SIZE)

# Split data
split = int(0.8 * len(all_texts))
train_texts, test_texts = all_texts[:split], all_texts[split:]
train_labels, test_labels = all_labels[:split], all_labels[split:]

print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

# Create dataloaders
train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_LEN)
test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model
model = SentimentLSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

print("\nTraining...")
print("=" * 50)

# Training loop
for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.float().to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += ((pred > 0).float() == y).sum().item()
        train_total += y.size(0)
    
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    # TEST
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.float().to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            test_loss += loss.item()
            test_correct += ((pred > 0).float() == y).sum().item()
            test_total += y.size(0)
    
    test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Acc: {test_acc*100:.2f}%")

print("=" * 50)
print("Training complete!")

# Save model
torch.save(model.state_dict(), 'sentiment_lstm.pth')
print("Model saved to sentiment_lstm.pth")

import torch
import torch.nn as nn
import torch.optim as optim
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ============================================
# CONFIG
# ============================================

VOCAB_SIZE = 10000 
EMBED_DIM = 128 
HIDDEN_SIZE = 256
NUM_LAYERS = 2
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ============================================
# TOKENIZER & VOCAB
# ============================================

def tokenize(text):
    """Convert text to tokens"""
    return re.sub(r"[^a-zA-Z ]", "", text.lower()).split()

def build_vocab(texts, max_vocab):
    """Build vocabulary from texts"""
    counter = Counter(t for text in texts for t in tokenize(text))
    vocab = {'<PAD>' : 0, '<UNK>' : 1}
    
    for word, _ in counter.most_common(max_vocab-2):
        vocab[word] = len(vocab)
    
    return vocab 

def encode(text, vocab, max_len):
    """Convert text to token IDs with padding"""
    ids = [vocab.get(t, 1) for t in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids)) 

# ============================================
# DATASET
# ============================================

class IMDBDataset(Dataset):
    """IMDB Dataset class"""
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
    """Bidirectional LSTM for sentiment classification"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (batch, seq_len)
        emb = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        out, (hn, _) = self.lstm(emb)           # out: (batch, seq_len, hidden*2)
        
        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden*2)
        
        return self.fc(self.dropout(hidden)).squeeze(1)  # (batch,)

# ============================================
# TRAINING & VALIDATION FUNCTIONS
# ============================================

def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # CRITICAL FOR LSTMs!
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        correct += ((pred > 0).float() == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, loader, loss_fn, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.float().to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            correct += ((pred > 0).float() == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ============================================
# DUMMY DATA (Replace with real IMDB data)
# ============================================

def create_dummy_data(num_samples=1000):
    """Create dummy IMDB-like data for demo"""
    positive_reviews = [
        "this movie was absolutely amazing and fantastic",
        "i loved every second of this film it was great",
        "best movie i have ever watched highly recommend",
        "incredible acting and wonderful story great movie",
        "outstanding performance and brilliant direction loved it"
    ]
    
    negative_reviews = [
        "this movie was terrible and a waste of time",
        "awful acting and boring story really bad film",
        "worst movie ever made absolutely horrible",
        "terrible performance and dull direction hated it",
        "boring and pointless waste of two hours"
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples // 2):
        texts.append(positive_reviews[i % len(positive_reviews)] + f" {i}")
        labels.append(1)  # positive
        
        texts.append(negative_reviews[i % len(negative_reviews)] + f" {i}")
        labels.append(0)  # negative
    
    return texts, labels

# ============================================
# MAIN TRAINING PIPELINE
# ============================================

print("=" * 60)
print("IMDB SENTIMENT ANALYSIS WITH BIDIRECTIONAL LSTM")
print("=" * 60)

# 1. CREATE DUMMY DATA (In real scenario, load IMDB dataset)
print("\n[1/5] Creating data...")
all_texts, all_labels = create_dummy_data(num_samples=2000)
print(f"Created {len(all_texts)} samples")

# 2. BUILD VOCABULARY
print("[2/5] Building vocabulary...")
vocab = build_vocab(all_texts, VOCAB_SIZE)
print(f"Vocabulary size: {len(vocab)}")

# 3. CREATE DATASETS & DATALOADERS
print("[3/5] Creating datasets...")
# Split data: 80% train, 20% test
split = int(0.8 * len(all_texts))
train_texts, test_texts = all_texts[:split], all_texts[split:]
train_labels, test_labels = all_labels[:split], all_labels[split:]

train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_LEN)
test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 4. CREATE MODEL, OPTIMIZER, LOSS
print("[4/5] Setting up model...")
model = SentimentLSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 5. TRAINING LOOP
print("[5/5] Training...")
print("=" * 60)

train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_val_acc = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
    
    # Validate
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, DEVICE)
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Save best model
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'sentiment_lstm_best.pth')
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
    print()

print("=" * 60)
print(f"Training complete!")
print(f"Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch+1})")
print("=" * 60)

# ============================================
# SAVE FINAL MODEL
# ============================================

torch.save(model.state_dict(), 'sentiment_lstm_final.pth')
print("\n✅ Model saved to 'sentiment_lstm_final.pth'")

# ============================================
# PLOTTING RESULTS
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
epochs_list = list(range(1, EPOCHS + 1))
axes[0].plot(epochs_list, train_losses, 'b-o', label='Train Loss', linewidth=2)
axes[0].plot(epochs_list, test_losses, 'r-o', label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[1].plot(epochs_list, [acc*100 for acc in train_accs], 'b-o', label='Train Accuracy', linewidth=2)
axes[1].plot(epochs_list, [acc*100 for acc in test_accs], 'r-o', label='Test Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✅ Graph saved to 'training_results.png'")
plt.show()

# ============================================
# INFERENCE / TESTING
# ============================================

def predict_sentiment(text, model, vocab, device, max_len=MAX_LEN):
    """Predict sentiment for a single text"""
    model.eval()
    
    with torch.no_grad():
        encoded = torch.tensor([encode(text, vocab, max_len)]).to(device)
        pred = model(encoded)
        score = torch.sigmoid(pred).item()  # Convert logits to probability
        
        sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
        confidence = score if score > 0.5 else 1 - score
        
    return sentiment, confidence

print("\n" + "=" * 60)
print("TESTING ON NEW REVIEWS")
print("=" * 60)

test_reviews = [
    "this movie was absolutely amazing and wonderful",
    "terrible film worst movie ever made",
    "great acting and brilliant story loved it",
    "boring waste of time awful performance"
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review, model, vocab, DEVICE)
    print(f"\nReview: \"{review}\"")
    print(f"Prediction: {sentiment} (confidence: {confidence*100:.2f}%)")

print("\n" + "=" * 60)
print("🔥 TRAINING COMPLETE - YOU BUILT A SENTIMENT CLASSIFIER! 🔥")
print("=" * 60)

# ============================================
# SUMMARY STATISTICS
# ============================================

print("\n📊 FINAL STATISTICS:")
print(f"  Initial Train Loss: {train_losses[0]:.4f}")
print(f"  Final Train Loss: {train_losses[-1]:.4f}")
print(f"  Loss Reduction: {(1 - train_losses[-1]/train_losses[0])*100:.2f}%")
print()
print(f"  Initial Train Acc: {train_accs[0]*100:.2f}%")
print(f"  Final Train Acc: {train_accs[-1]*100:.2f}%")
print(f"  Best Test Acc: {best_val_acc*100:.2f}%")
print()
print(f"  Total Parameters: {total_params:,}")
print(f"  Training Time: ~{EPOCHS * 2:.0f} minutes (estimated)")
print("=" * 60)
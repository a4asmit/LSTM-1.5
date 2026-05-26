import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=False
)

x = torch.randn(32, 50, 10)
h0 = torch.zeros(2, 32, 64)
c0 = torch.zeros(2, 32, 64)

output, (cn, hn) = lstm(x, (h0, c0))
"""
print(output.shape)
print(hn.shape)
print(cn.shape)
"""

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        output, (hn, cn) = self.lstm(emb)
        last = hn[-1]
        return self.fc(self.dropout(last))
    
model = LSTMClassifier(vocab_size=10000, embed_dim=128, hidden_size=256, num_layers=2, num_classes=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
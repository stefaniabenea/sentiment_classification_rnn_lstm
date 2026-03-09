import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_classes,num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_dim,batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self,x):
        # L=sequence length
        # (B, L) -> (B, L, embed_dim)
        embedded = self.embedding(x)
        # output: (B, L, hidden_dim)
        output, hidden = self.rnn(embedded)
        # (num_layers, B, hidden_dim) -> (B, hidden_dim)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return out
    
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self,x):
        # (B, L) -> (B,L,embed_size)
        embedded = self.embedding(x)
        # output: (B,L,hidden_size)
        output, (hidden, cell) = self.lstm(embedded)
        # (num_layers, B, hidden_dim) -> (B, hidden_dim)
        hidden = hidden[-1]
        # (B, num_classes)
        out = self.fc(hidden)
        return out

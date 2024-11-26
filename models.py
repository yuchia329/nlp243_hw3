import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, embedding_dim, embedding_matrix, dropout, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding =  nn.Sequential(
            nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, device=device), freeze=False),  # Pretrained GloVe
            nn.Linear(embedding_dim, d_model).to(device),  # Projection to d_model
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.embedding_layer_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.embedding_layer_norm(x)
        x = self.dropout(x)
        
        seq_length = x.size(0)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.fc(x)
        return logits

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embedding_matrix=None):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
            self.embedding.weight.requires_grad = True  # Allow fine-tuning
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        lstm_out, hidden_state = self.lstm(embedded, hidden_state)  # [batch_size, seq_length, hidden_dim]
        logits = self.fc(lstm_out)  # [batch_size, seq_length, vocab_size]
        return logits, hidden_state

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embedding_matrix=None):
        super(GRUModel, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
            self.embedding.weight.requires_grad = True  # Allow fine-tuning
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        gru_out, _ = self.gru(embedded, hidden_state)  # [batch_size, seq_length, hidden_dim]
        logits = self.fc(gru_out)  # [batch_size, seq_length, vocab_size]
        return logits, hidden_state
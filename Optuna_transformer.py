import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from functools import partial
import os
import random
import pandas as pd
import re
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import gensim.downloader as api

# Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.reverse_vocab = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}
        
    def fit(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                word = word.lower()
                word = re.sub(r'[^\w\s]', '', word)
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.reverse_vocab[idx] = word
    
    def encode(self, sentence, max_length):
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        if len(token_ids) < max_length:
            token_ids += [self.vocab["<pad>"]] * (max_length - len(token_ids))
        return token_ids[:max_length]
    
    def decode(self, token_ids):
        return [self.reverse_vocab.get(idx, "<unk>") for idx in token_ids]

# Dataset
class NgramDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [tokenizer.encode(text, max_length) for text in texts]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return torch.tensor(input_ids), torch.tensor(target_ids)

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, embedding_dim, embedding_matrix, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding =  nn.Sequential(
            nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, device=device), freeze=False),  # Pretrained GloVe
            nn.Linear(embedding_dim, d_model).to(device),  # Projection to d_model
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        seq_length = x.size(0)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.fc(x)
        return logits

def loadGlove(tokenizer):
    glove = api.load("glove-wiki-gigaword-100")
    embedding_dim = 100  # Dimension of GloVe vectors
    vocab_size = len(tokenizer.vocab)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    for word, idx in tokenizer.vocab.items():
        if word in glove:  # Check if the word exists in GloVe
            embedding_matrix[idx] = glove[word]
    return embedding_matrix

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # Mask out the padding tokens by setting them to ignore_index
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            target = target.clone()  # Avoid modifying the original target
            target[pad_mask] = 0  # Set pad indices to 0 for smoothing compatibility

        # Compute smoothed targets
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        n_classes = logits.size(-1)
        one_hot = torch.zeros_like(log_prob).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Mask the loss contributions for padding tokens
        loss = -torch.sum(log_prob * smooth_target, dim=-1)
        if self.ignore_index is not None:
            loss[pad_mask] = 0.0

        # Average the loss over valid tokens
        return loss.sum() / (~pad_mask).sum()


from torch.nn.utils.rnn import pad_sequence

# Collate function for padding
def collate_fn(batch):
    # Pad all sequences in the batch to the length of the longest sequence
    batch = [torch.tensor(sample) for sample in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=pad_idx)  # Use pad_idx for padding
    return batch


# Training and evaluation function for Optuna
def train_and_evaluate(trial, tokenizer, max_seq_length, embedding_dim, embedding_matrix, train_texts, test_texts, device):
    # Hyperparameters to tune
    
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs =15 # trial.suggest_int("epochs", 20, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
    num_layers = trial.suggest_int("num_layers", 2, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Datasets and DataLoaders
    train_dataset = NgramDataset(train_texts, tokenizer, max_seq_length)
    test_dataset = NgramDataset(test_texts, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model
    model = TransformerModel(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers, 
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        dropout=dropout,
    ).to(device)
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(ignore_index=tokenizer.vocab["<pad>"],smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, target_ids in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            # padding_mask = (input_ids == tokenizer.vocab["<pad>"])
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluation
    model.eval()
    total_perplexity = 0
    sentence_count = 0
    with torch.no_grad():
        for input_ids, target_ids in test_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits = model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            
            for i, target_seq in enumerate(target_ids):
                prob_seq = probabilities[i, torch.arange(target_seq.size(0)), target_seq]
                valid_probs = prob_seq[target_seq != pad_idx]
                if len(valid_probs) > 0:
                    perplexity = torch.exp(-torch.mean(torch.log(valid_probs)))
                    total_perplexity += perplexity.item()
                    sentence_count += 1
    
    # Average perplexity
    avg_perplexity = total_perplexity / sentence_count
    print('avg_perplexity: ', avg_perplexity)
    return avg_perplexity

# Optuna optimization
def objective(trial, tokenizer, max_seq_length, embedding_dim, embedding_matrix, train_texts, test_texts, device):
    return train_and_evaluate(trial, tokenizer, max_seq_length, embedding_dim, embedding_matrix, train_texts, test_texts, device)



seed = 490
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Data
ptb = load_dataset('ptb-text-only/ptb_text_only') # download the dataset
train_texts = [item["sentence"] for item in  ptb['train']]
test_texts = [item["sentence"] for item in  ptb['test']]
train_large, train_small = train_test_split(
        train_texts, 
        test_size=0.2, 
        random_state=490
    )
test_large, test_small = train_test_split(
    test_texts, 
    test_size=0.2, 
    random_state=490
)

# Hyperparameters
max_seq_length = max(len(text.split()) for text in train_texts) + 2
embedding_dim=100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer and dataset
tokenizer = SimpleTokenizer()
tokenizer.fit(train_texts)# + test_texts)

embedding_matrix = loadGlove(tokenizer)
vocab = tokenizer.vocab
pad_idx = vocab["<pad>"]

# Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(partial(objective, tokenizer=tokenizer, max_seq_length=max_seq_length, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix, train_texts=train_small, test_texts=test_small, device=device), n_trials=20)

print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")

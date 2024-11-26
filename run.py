import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
import os
import random
import pandas as pd
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
import gensim.downloader as api
from collections import Counter
from models import TransformerModel, LSTMModel, GRUModel
import argparse

def get_word_frequencies(sentences):
    word_counter = Counter()
    for sentence in sentences:
        words = sentence.split()  # Tokenize the sentence into words
        for word in words:
            word = word.lower()
            word = re.sub(r'[^\w\s]', '', word)
            word_counter.update(words)
    return word_counter

# Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.reverse_vocab = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}
        
    def fit(self, sentences):
        min_freq = 5
        word_frequencies = get_word_frequencies(sentences)
        for sentence in sentences:
            for word in sentence.split():
                word = word.lower()
                word = re.sub(r'[^\w\s]', '', word)
                if word not in self.vocab:
                    if word_frequencies[word] > min_freq:
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

# Training
def train_model(model, dataloader, criterion, optimizer, epochs, device):
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, target_ids in dataloader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        if scheduler:
            scheduler.step()

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    sentence_perplexities = []
    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits = model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            
            for i, target_seq in enumerate(target_ids):
                # Calculate the probabilities of the target tokens
                prob_seq = probabilities[i, torch.arange(target_seq.size(0)), target_seq]
                
                # Avoid log(0) by masking zero probabilities (e.g., due to padding)
                valid_probs = prob_seq[target_seq != tokenizer.vocab["<pad>"]]
                
                if len(valid_probs) > 0:  # Only calculate perplexity if there are valid tokens
                    perplexity = torch.exp(-torch.mean(torch.log(valid_probs)))
                else:
                    perplexity = torch.tensor(float('inf'))  # For empty sequences, set high perplexity
                
                sentence_perplexities.append({"ID":batch_idx * dataloader.batch_size + i, "ppl":perplexity.item()})
    
    # Print results
    return sentence_perplexities
    
def loadGlove(tokenizer):
    glove = api.load("glove-wiki-gigaword-100")
    embedding_dim = 100  # Dimension of GloVe vectors
    vocab_size = len(tokenizer.vocab)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    for word, idx in tokenizer.vocab.items():
        if word in glove:  # Check if the word exists in GloVe
            embedding_matrix[idx] = glove[word]
    return embedding_matrix

def select_hyperparam(name):
    if name == "transformer":
        params = {
            "epochs": 10,
            "lr": 0.000207,
            "dropout": 0.33,
            "num_layers": 4,
            "embedding_dim": 100,
            "d_model": 64,
            "nhead": 8,
        } 
    elif name == "LSTM":
        params = {
            "epochs": 10,
            "lr": 0.0035,
            "dropout": 0.409875,
            "num_layers": 3,
            "embedding_dim": 100,
            "hidden_dim": 128,
        }
    else:
        params = {
            "epochs": 10,
            "lr": 0.0010091900558751748,
            "dropout": 0.18881316661847183,
            "num_layers": 3,
            "embedding_dim": 100,
            "hidden_dim": 256,
        }
    return params
        
def select_model(name):
    if name == "transformer":
        params = select_hyperparam(name)
        model = TransformerModel(vocab_size=len(tokenizer.vocab),
            num_layers=params["num_layers"], 
            max_seq_length=max_seq_length,
            embedding_dim=params["embedding_dim"],
            dropout=params["dropout"],
            embedding_matrix=embedding_matrix,
            d_model=params["d_model"],
            nhead=params["nhead"],
        )
    elif name == "LSTM":
        params = select_hyperparam(name)
        model = LSTMModel(
            vocab_size=len(tokenizer.vocab),
            embedding_dim=params["embedding_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            embedding_matrix=embedding_matrix,
            hidden_dim=params["hidden_dim"],
        )
    else:
        params = select_hyperparam(name)
        model = GRUModel(
            vocab_size=len(tokenizer.vocab),
            embedding_dim=params["embedding_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            embedding_matrix=embedding_matrix,
            hidden_dim=params["hidden_dim"],
        )
    return model, params["lr"]

def seed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a file.")
    parser.add_argument("output", help="Path to the output file")
    args = parser.parse_args()
    output_file = args.output
    
    seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    ptb = load_dataset('ptb-text-only/ptb_text_only')
    train_texts = [item["sentence"] for item in  ptb['train']]
    test_texts = [item["sentence"] for item in  ptb['test']]

    # Hyperparameters
    max_seq_length = max(len(text.split()) for text in train_texts) + 2
    batch_size = 32
    epochs = 10

    # Tokenizer and dataset
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_texts)
    train_dataset = NgramDataset(train_texts, tokenizer, max_seq_length)
    test_dataset = NgramDataset(test_texts, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # embedding
    embedding_matrix = loadGlove(tokenizer)
    
    # Model, loss, and optimizer
    model, lr = select_model("transformer")
    model = model.to(device)
    criterion = LabelSmoothingLoss(ignore_index=tokenizer.vocab["<pad>"],smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training and evaluation
    train_model(model, train_loader, criterion, optimizer, epochs, device)
    result = evaluate_model(model, test_loader, tokenizer, device)
    df = pd.DataFrame(result)
    print('score: ', df[['ppl']].mean(axis=1))
    df.to_csv(output_file, index=False)


!pip install wandb

!wandb login PUT_YOUR_API_KEY

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Mount files
from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""# Data processing"""

def data_init():
    with open("/content/drive/MyDrive/Colab Notebooks/data/ptb.train.txt") as f:
        train = f.read().strip().replace('\n', '<eos>').split()
    with open("/content/drive/MyDrive/Colab Notebooks/data/ptb.valid.txt") as f:
        val = f.read().strip().replace('\n', '<eos>').split()
    with open("/content/drive/MyDrive/Colab Notebooks/data/ptb.test.txt") as f:
        test = f.read().strip().replace('\n', '<eos>').split()

    words = sorted(set(train))
    word2idx = {word: idx for idx, word in enumerate(words)}
    trn = [word2idx[w] for w in train]
    vld = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in val]
    tst = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in test]

    return np.array(trn), np.array(vld), np.array(tst), len(words)

train_set, val_set, test_set, vocab_size = data_init()

print("Train set shape:", train_set.shape)
print("Validation set shape:", val_set.shape)
print("Test set shape:", test_set.shape)
print("Vocabulary size:", vocab_size)
print(train_set[:20])
print(train_set[20:40])

# print(len(train_text), train_text[:10])
# print(len(valid_text), valid_text[:10])
# print(len(test_text), test_text[:10])

# Batch data preparation
def minibatch(data, batch_size, seq_length):
    data = torch.tensor(data, dtype=torch.int64)
    num_batches = data.size(0) // batch_size
    data = data[:num_batches * batch_size].view(batch_size, -1)

    dataset = []
    for i in range(0, data.size(1) - seq_length+1, seq_length):
        x = data[:, i:i + seq_length].transpose(1, 0)
        y = data[:, i+1:i+seq_length+1].transpose(1, 0)
        dataset.append((x, y))
    return dataset

#Testing minibatch
batch_size = 20
seq_length = 20

train_batch = minibatch(train_set, batch_size, seq_length)
valid_batch = minibatch(val_set, batch_size, seq_length)
test_batch = minibatch(test_set, batch_size, seq_length)

print(len(train_batch))
print(len(valid_batch))
print(len(test_batch))

print(train_batch[0][0].shape)
print(train_batch[0][1].shape)
print(train_batch[0][0])
print(train_batch[0][1])
print("*********")

# for i, (x, y) in enumerate(valid_batch):
#     print(f"Batch {i}: x shape: {x.shape}, y shape: {y.shape}")

"""# Defining our models"""

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout, rnn_type='LSTM'):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Embedding layer to map input tokens to vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # RNN layer (either LSTM or GRU based on user choice)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Linear layer to map from hidden state to vocabulary size (for logits)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.init_parameters()

    # Initialize parameters to U(-0.1, 0.1)
    def init_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    # Forward pass: directly from paper
    def forward(self, x, states):
        x = self.dropout(self.embedding(x))  # Embedding input, then dropout
        x, states = self.rnn(x, states)  # Pass through RNN (LSTM or GRU)
        x = self.dropout(x)  # Apply dropout after rnn again
        x = self.fc(x)  # Final fully connected layer to get logits
        return x, states

    # Initialize hidden (and cell) states
    def state_init(self, batch_size):
        if self.rnn_type == 'LSTM':
            # h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            # c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

            # h0 = torch.nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, self.hidden_size)).to(device)
            # c0 = torch.nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, self.hidden_size)).to(device)

            return (h0, c0)
        else:  # GRU has only hidden states
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            # h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            # h0 = torch.nn.init.xavier_uniform_(torch.empty(self.num_layers, batch_size, self.hidden_size)).to(device)

            return h0

    # Detach hidden states (to avoid backpropagating through entire sequence)
    def detach(self, states):
        if isinstance(states, tuple):  # LSTM states
            return (states[0].detach(), states[1].detach())
        else:  # GRU state
            return states.detach()

# Perplexity calculation
def perplexity(data, model, batch_size):
    with torch.no_grad():
        losses = []
        states = model.state_init(batch_size)
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            scores, states = model(x, states)
            loss = cross_entropy_loss(scores, y)
            losses.append(loss.item())
    return np.exp(np.mean(losses))

# Cross-entropy loss function
def cross_entropy_loss(scores, y):
    criterion = nn.CrossEntropyLoss()
    scores = scores.reshape(-1, scores.size(2))
    y = y.reshape(-1)
    loss = criterion(scores, y)
    return loss

import timeit

def train(data, model, epochs, initial_learning_rate, id, max_grad_norm, epoch_threshold, lr_decay, step_size=6, gamma=1.0/1.65, dropout=False):
    wandb.init(
        project="dl-ex2",
        name=f'{model.rnn_type}_lr_{initial_learning_rate}_dropout_{model.dropout.p}',
        config={
        "learning_rate": initial_learning_rate,
        "architecture": model.rnn_type,
        "hidden_size": model.hidden_size,
        "layer_num": model.num_layers,
        "epochs": epochs,
        "dropout": model.dropout.p,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "max_grad_norm": max_grad_norm
        }
    )

    trn, vld, tst = data
    tic = timeit.default_timer()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        states = model.state_init(batch_size)
        total_loss = 0.0
        total_words = 0

        for i, (x, y) in enumerate(trn):
            x = x.to(device)
            y = y.to(device)

            states = model.detach(states)
            optimizer.zero_grad()

            # Forward pass
            scores, states = model(x, states)

            # Loss and Backpropagation
            loss = cross_entropy_loss(scores, y)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_words += y.numel()
            # Print 10 times per batch
            if i % (len(trn)//10) == 0:
                toc = timeit.default_timer()
                print("batch no = {:d} / {:d}, ".format(i, len(trn)) +
                      # "avg train loss per word this batch = {:.3f}, ".format(loss.item()/(y.numel())) +
                      "avg train loss per word this batch = {:.3f}, ".format(loss.item()) +
                      "words per second = {:d}, ".format(round(total_words/(toc-tic))) +
                      "lr = {:.3f}, ".format(optimizer.param_groups[0]['lr']) +
                      "since beginning = {:d} mins, ".format(round((toc-tic)/60)))

        avg_train_loss = total_loss / len(trn)
        train_perp = perplexity(trn, model, batch_size)

        # Validation and Test perplexity
        model.eval()
        val_perp = perplexity(vld, model, batch_size)
        test_perp = perplexity(tst, model, batch_size)
        print(f"Epoch {epoch + 1}: Start Learning Rate: {initial_learning_rate}, Dropout: {model.dropout.p}")
        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.3f}")
        print(f"Epoch {epoch + 1}: Train Perplexity: {train_perp:.3f}")
        print(f"Epoch {epoch + 1}: Validation Perplexity: {val_perp:.3f}")
        print(f"Epoch {epoch + 1}: Test Perplexity: {test_perp:.3f}")

        # Wandb Plotting
        wandb.log({"Train Perplexity": train_perp, "Validation Perplexity": val_perp, "Test Perplexity": test_perp, "epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr'],"dropout": model.dropout.p })

        # 1. Use scheduler to decay LR every # of steps
        scheduler.step()

        # 2. Paper -> Decay LR every step after a certain threshold
        # if epoch >= epoch_threshold:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= lr_decay  # Decay learning rate

        # Save the best model
        if val_perp < best_val_loss:
            print(f"Saw better model at Epoch {epoch+1}")
            best_val_loss = val_perp
            best_model = {k: v.clone() for k, v in model.state_dict().items()}

    # Test set perplexity
    model.load_state_dict(best_model)
    test_perp = perplexity(tst, model, batch_size)
    print(f"Test Set Perplexity: {test_perp:.3f} Model: {model.rnn_type} Dropout: {model.dropout.p} Hidden_size: {model.hidden_size}")

    torch.save(best_model, 'best_model.pth')
    print("Training complete. Best model saved.")

"""# Training"""

# Hyperparameters
batch_size = 20
seq_length = 20
hidden_size = 200
layer_num = 2
max_grad_norm = 5

total_epochs = 13
dropout = 0.0

# only used for paper implentation of decay after threshold
epoch_threshold = 7
lr_decay = 0.5

# Initialize datasets
trn, vld, tst, vocab_size = data_init()

trn = minibatch(trn, batch_size, seq_length)
vld = minibatch(vld, batch_size, seq_length)
tst = minibatch(tst, batch_size, seq_length)

def run_experiments_no_dropout():

    dropout = 0.0
    total_epochs = 15
    learning_rate = 2.0
    step_size = 5
    gamma = 0.5

    for rnn_type in ['LSTM', 'GRU']:
        model = Model(vocab_size, hidden_size, layer_num, dropout, rnn_type=rnn_type).to(device)
        train((trn, vld, tst), model, total_epochs, learning_rate, max_grad_norm, epoch_threshold, lr_decay, step_size, gamma, dropout=False)

def run_experiments_with_dropout():

    total_epochs = 25
    step_size = 6

    for dropout in [0.25]:
      for rnn_type in ['LSTM', 'GRU']:
          model = Model(vocab_size, hidden_size, layer_num, dropout, rnn_type=rnn_type).to(device)
          learning_rate = 4.0 if rnn_type=='LSTM' else 2.0
          gamma = 1.0/1.65 if rnn_type == 'GRU' else 1.0/1.15
          train((trn, vld, tst), model, total_epochs, learning_rate, max_grad_norm, epoch_threshold, lr_decay, step_size, gamma, dropout=True)

# Call the function to run experiments
run_experiments_no_dropout()
run_experiments_with_dropout()

"""# Table"""

import pandas as pd
import plotly.graph_objects as go

# Perplexity values are from WandB runs history
data = {
    'Model': ['LSTM No Dropout', 'GRU No Dropout', 'LSTM 25% Dropout', 'GRU 25% Dropout'],
    'Training Perplexity': [72.49, 67.35, 67.97, 59.98],
    'Validation Perplexity': [123.75, 124.40, 102.66, 104.49],
    'Test Perplexity': [119.82, 119.98, 99.08, 100.88]
}

# Display table
df = pd.DataFrame(data)

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='coral',
                align='left'),
    cells=dict(values=[df[col] for col in df.columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(
    title="Perplexities of Various Models",
    height=500,
    width=750
)

fig.show()
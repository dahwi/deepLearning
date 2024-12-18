# -*- coding: utf-8 -*-
"""vae.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ONAEttmZsNFzRQJwgvdymWo_7mEDNivK
"""

!pip install wandb

!wandb login 9172fb113e07d174f618e9042047cc5c4adacc0f

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import wandb
from sklearn.externals import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

manualSeed = 42
torch.manual_seed(manualSeed)
# Sets the seed for generating random numbers for all devices (CPU and CUDA).
# Ensures that the same random numbers are generated each time the code is run,
# making the results reproducible.

torch.use_deterministic_algorithms(True)
# Enables deterministic algorithms for operations in PyTorch. This means that
# the operations will always produce the same output given the same inputs,
# even on different hardware or different runs.
# It is crucial for reproducibility, particularly when using CUDA.

def load_fashion_mnist(batch_size=128):
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor()])

    # Load Fashion MNIST
    # TODO: change to read from local
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def get_label_subset(train_dataset, num_labels, seed=42):
    np.random.seed(seed)  # Set the seed for reproducibility
    indices = []
    num_classes = 10
    examples_per_class = num_labels // num_classes

    for label in range(num_classes):
        label_indices = np.where(np.array(train_dataset.targets) == label)[0]
        np.random.shuffle(label_indices)  # Shuffle the indices
        indices += list(label_indices[:examples_per_class])
    return Subset(train_dataset, indices)

train_loader, test_loader, train_dataset, test_dataset = load_fashion_mnist()

import matplotlib.pyplot as plt

# Assuming you have a 2D array called 'image_array' representing the pixel data

def display_image(image_array):
  plt.imshow(image_array, cmap='gray')
  plt.show()


# Example usage:
print(train_dataset.data.shape)
print(train_dataset[0][0].shape)
print(train_dataset[0][1])
print(train_dataset.targets.shape)
image_array = train_dataset[0][0].numpy().squeeze()  # Accessing the image data from the dataset
display_image(image_array)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=600, latent_dim=50):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # encoder produces mean and log of variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.2)
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        var = torch.exp(0.5 * logvar)
        # sampling epsilon
        eps = torch.randn_like(var).to(device)
        return mu + eps * var

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z), negative_slope=0.2)
        h4 = F.leaky_relu(self.fc4(h3), negative_slope=0.2)
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(epochs, learning_rate, data_loader):
  vae = VAE().to(device)
  optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

  wandb.init(
    project="dl-ex3",
    name=f'vae_lr_{learning_rate}_epochs_{epochs}',
    config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    }
  )

  vae.train()
  for i in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
      data = data.view(-1, 784).to(device)
      optimizer.zero_grad()
      recon_batch, mu, logvar = vae(data)
      loss = vae_loss(recon_batch, data, mu, logvar)
      loss.backward()
      train_loss += loss.item()
      optimizer.step()

    avg_loss = train_loss / len(data_loader.dataset)
    wandb.log({"Train Loss": avg_loss, "epoch": i})     # Wandb Plotting
    print("\tEpoch", i + 1, "complete!", "\tAverage Loss: ", avg_loss)
  wandb.finish()

  return vae

def extract_latent_features(model, data_loader):
    model.eval()
    latent_features = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(-1, 784).to(device)
            target = target.to(device)
            mu, _ = model.encode(data)
            latent_features.append(mu.cpu().numpy())
            labels.append(target.cpu().numpy())

    return np.vstack(latent_features), np.hstack(labels)

def train_svm(train_latent, train_labels, kernel='rbf'):
    scaler = StandardScaler()
    train_latent = scaler.fit_transform(train_latent)

    svm = SVC(kernel=kernel)
    svm.fit(train_latent, train_labels)

    return svm, scaler

def test_svm(svm, scaler, test_latent, test_labels):
    test_latent = scaler.transform(test_latent)
    predictions = svm.predict(test_latent)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy * 100.0

def run_experiment():

  train_loader, test_loader, train_dataset, test_dataset = load_fashion_mnist()
  epochs = 50
  learning_rate = 1e-3
  accuracies = {}
  for n in [100, 600, 1000, 3000]:
    train_subset_loader = DataLoader(get_label_subset(train_dataset, n), batch_size=128, shuffle=True)

    vae = train_vae(100 if n == 100 else epochs, learning_rate, train_subset_loader)
    torch.save(vae, f'/content/drive/MyDrive/Colab Notebooks/model/vae_model_w_{n}_samples.pth')
    train_latent, train_labels = extract_latent_features(vae, train_subset_loader)
    test_latent, test_labels = extract_latent_features(vae, test_loader)

    svm, scaler = train_svm(train_latent, train_labels, kernel='rbf')
    joblib.dump(svm, f'/content/drive/MyDrive/Colab Notebooks/model/svm_model_{n}_rbf.pkl')
    accuracy = test_svm(svm, scaler, test_latent, test_labels)
    accuracies[n] = accuracy
  return accuracies
final_accuracies = run_experiment()
final_accuracies
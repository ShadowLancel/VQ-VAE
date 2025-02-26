import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader

# Гиперпараметры
latent_dim = 64
num_embeddings = 1024
commitment_cost = 0.25
target_sr = 44100
duration = 4
batch_size = 16
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform

# Датасет с генератором
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform = load_audio(self.files[idx])
        return waveform

# VQ-VAE модель (simple VQ-VAE)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        x_flat = x.permute(0, 2, 1).reshape(-1, self.embedding_dim)  # [B*T, D]
        distances = (torch.sum(x_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(x_flat, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(x.shape)
        loss = F.mse_loss(quantized.detach(), x) + self.commitment_cost * F.mse_loss(quantized, x.detach())
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=latent_dim, num_embeddings=num_embeddings, commitment_cost=commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)
        z_q = F.normalize(z_q, dim=-1)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, z_q

def compute_cosine_similarity(original, reconstructed):
    original = original.flatten(start_dim=1)  # [B, T]
    reconstructed = reconstructed.flatten(start_dim=1)  # [B, T]
    return F.cosine_similarity(original, reconstructed, dim=1).mean().item()

def validate(model, val_loader, criterion):
    model.eval()
    val_loss, val_cosine = 0, 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_recon, vq_loss, _ = model(x)
            loss = criterion(x_recon, x) + vq_loss
            val_loss += loss.item()
            cosine_sim = compute_cosine_similarity(x, x_recon)
            val_cosine += cosine_sim
    return val_loss / len(val_loader), val_cosine / len(val_loader)

model = VQVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
dataset = AudioDataset(r'E:\C418_all')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0

    def check(self, loss):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=3)

for epoch in range(20):
    model.train()
    train_loss, train_cosine = 0, 0
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, vq_loss, _ = model(x)
        loss = F.mse_loss(x_recon, x) + vq_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_cosine += compute_cosine_similarity(x, x_recon)
    val_loss, val_cosine = validate(model, val_loader, F.mse_loss)
    scheduler.step()
    if early_stopping.check(val_loss):
        print(f"Early stopping на {epoch + 1} эпохе.")
        break
    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, Train Cosine: {train_cosine/len(train_loader):.6f}")
    print(f"           Val Loss: {val_loss:.6f}, Val Cosine: {val_cosine:.6f}")
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'vqvae_model_{epoch}.pth')
torch.save(model.state_dict(), 'final_vqvae_model.pth')
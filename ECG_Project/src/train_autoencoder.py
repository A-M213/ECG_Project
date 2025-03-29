import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.autoencoder import ECGDenoisingAutoencoder
from data.ECGDataset import ECGDataset
import os
import numpy as np

def train_autoencoder(
    batch_size=32,
    epochs=20,
    lr=1e-3,
    model_save_path="models/autoencoder.pth",
    device=None
):
    # Choix du device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Entraînement sur : {device}")

    # Chargement des données
    train_dataset = ECGDataset(
        path_noisy="data/noisy/X_train_noisy.npy",
        path_clean="data/processed/X_train.npy"
    )
    val_dataset = ECGDataset(
        path_noisy="data/noisy/X_val_noisy.npy",
        path_clean="data/processed/X_val.npy"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialisation du modèle
    model = ECGDenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Suivi des pertes
    train_losses = []
    val_losses = []

    print("🚀 Démarrage de l'entraînement...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Évaluation validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"📚 Époch {epoch}/{epochs} | Loss train : {train_loss:.6f} | val : {val_loss:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Sauvegarde modèle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modèle sauvegardé dans : {model_save_path}")

    # Sauvegarde des courbes de pertes
    os.makedirs("results", exist_ok=True)
    np.save("results/train_losses.npy", np.array(train_losses))
    np.save("results/val_losses.npy", np.array(val_losses))
    print("📈 Courbes de pertes sauvegardées dans : results/")
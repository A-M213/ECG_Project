
import torch
import numpy as np
from models.diffusion import Simple1DUNet, GaussianDiffusion1D
import os

def train_diffusion_model(
    path_clean,
    model_save_path,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    timesteps=200,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Entraînement sur : {device}")

    # Chargement des signaux propres
    #X_train = np.load(path_clean).astype(np.float32)
    X_train = np.load(path_clean, mmap_mode="r")[:1000].astype(np.float32)
    print(f"📥 Données chargées : {X_train.shape}")

    # Initialisation du modèle
    model = Simple1DUNet().to(device)
    diffusion = GaussianDiffusion1D(model, timesteps=timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Entraînement
    print("🚀 Début de l'entraînement DDPM...")
    for epoch in range(epochs):
        losses = []
        model.train()
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            xb = torch.tensor(xb.transpose(0, 2, 1)).to(device)
            t = torch.randint(0, diffusion.timesteps, (xb.size(0),), device=device).long()

            try:
                loss = diffusion.p_losses(xb, t)
            except Exception as e:
                print("🔥 Erreur rencontrée dans diffusion.p_losses")
                print(f"→ xb.shape : {xb.shape}")
                print(f"→ t.shape  : {t.shape}")
                raise e
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"📚 Époch {epoch+1}/{epochs} – Loss: {np.mean(losses):.6f}")

    # Sauvegarde
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modèle sauvegardé dans : {model_save_path}")

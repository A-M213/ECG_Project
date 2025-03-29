import torch
import torch.nn as nn
import numpy as np
from models.autoencoder import ECGDenoisingAutoencoder
from data.ECGDataset import ECGDataset
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import os

def compute_snr(clean, denoised):
    noise = clean - denoised
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / (power_noise + 1e-8))

def evaluate_autoencoder(
    model_path,
    path_noisy,
    path_clean,
    path_labels,
    batch_size=32,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Évaluation sur : {device}")

    # Chargement du modèle
    model = ECGDenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Chargement des données
    dataset = ECGDataset(path_noisy, path_clean)
    loader = DataLoader(dataset, batch_size=batch_size)

    labels = np.load(path_labels)
    denoised_all = []
    clean_all = []

    print("🔍 Prédiction des signaux...")
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            denoised_all.append(preds)
            clean_all.append(yb.numpy())

    denoised_all = np.concatenate(denoised_all, axis=0)
    clean_all = np.concatenate(clean_all, axis=0)

    # Remettre [C, T] → [T, C]
    denoised_all = np.transpose(denoised_all, (0, 2, 1))
    clean_all = np.transpose(clean_all, (0, 2, 1))

    print("📊 Calcul des métriques...")
    mse_total = np.mean((denoised_all - clean_all) ** 2)
    snr_total = np.mean([compute_snr(c, d) for c, d in zip(clean_all, denoised_all)])
    corr_total = np.mean([
        np.mean([pearsonr(c[:, i], d[:, i])[0] for i in range(12)])
        for c, d in zip(clean_all, denoised_all)
    ])

    print(f"\n📈 Résultats globaux sur X_test :")
    print(f"   - MSE  : {mse_total:.6f}")
    print(f"   - SNR  : {snr_total:.2f} dB")
    print(f"   - Corr : {corr_total:.4f}")

    print("\n📊 Résultats par type de bruit :")
    types_bruit = ["gaussian", "baseline", "powerline", "motion"]
    for bruit in types_bruit:
        indices = np.where(labels == bruit)[0]
        c_subset = clean_all[indices]
        d_subset = denoised_all[indices]
        mse = np.mean((d_subset - c_subset) ** 2)
        snr = np.mean([compute_snr(c, d) for c, d in zip(c_subset, d_subset)])
        corr = np.mean([
            np.mean([pearsonr(c[:, i], d[:, i])[0] for i in range(12)])
            for c, d in zip(c_subset, d_subset)
        ])
        print(f"   [{bruit}]  MSE: {mse:.6f} | SNR: {snr:.2f} dB | Corr: {corr:.4f}")

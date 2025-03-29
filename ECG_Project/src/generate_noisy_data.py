import numpy as np
from src.noise import (
    add_gaussian_noise,
    add_baseline_wander,
    add_powerline_noise,
    add_motion_artifact
)

def process_dataset(X, set_name="train", snr_gauss=12, snr_powerline=18, snr_motion=12):
    print(f"\n📥 Traitement du set : {set_name.upper()}")
    n = X.shape[0]
    quarter = n // 4
    remainder = n % 4
    print(f"→ Total signaux : {n}")
    print(f"→ Répartition : {quarter} par bruit, reste : {remainder}")

    X_noisy = np.zeros_like(X)
    noise_labels = []

    print("🔊 1. Bruit gaussien...")
    for i in range(quarter):
        X_noisy[i] = add_gaussian_noise(X[i], snr_db=snr_gauss)
        noise_labels.append("gaussian")
        if i % 50 == 0: print(f"   → {i}/{quarter}")

    print("🌊 2. Baseline wander...")
    for i in range(quarter, 2*quarter):
        X_noisy[i] = add_baseline_wander(X[i])
        noise_labels.append("baseline")
        if i % 50 == 0: print(f"   → {i - quarter}/{quarter}")

    print("⚡ 3. Bruit secteur (50 Hz)...")
    for i in range(2*quarter, 3*quarter):
        X_noisy[i] = add_powerline_noise(X[i], snr_db=snr_powerline)
        noise_labels.append("powerline")
        if i % 50 == 0: print(f"   → {i - 2*quarter}/{quarter}")

    print("📉 4. Mouvement électrode...")
    for i in range(3*quarter, n):
        X_noisy[i] = add_motion_artifact(X[i], snr_db=snr_motion)
        noise_labels.append("motion")
        if i % 50 == 0: print(f"   → {i - 3*quarter}/{quarter + remainder}")

    # Sauvegarde
    np.save(f"data/noisy/X_{set_name}_noisy.npy", X_noisy)
    np.save(f"data/noisy/X_{set_name}_noisy_labels.npy", np.array(noise_labels))
    print(f"✅ X_{set_name}_noisy.npy et labels sauvegardés dans data/noisy/")

def generate_noisy_data():
    print("📦 Génération des versions bruitées...")

    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    X_test = np.load("data/processed/X_test.npy")

    process_dataset(X_train, set_name="train")
    process_dataset(X_val, set_name="val")
    process_dataset(X_test, set_name="test")

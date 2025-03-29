import numpy as np

def calculate_noise_power(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    return noise_power

def add_gaussian_noise(signal, snr_db=12):
    """Ajoute un bruit gaussien blanc (EMG-like)."""
    noise_power = calculate_noise_power(signal, snr_db)
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_baseline_wander(signal, freq=0.33, amplitude=0.1):
    """Ajoute une dérive de ligne de base (baseline wander)."""
    n_samples = signal.shape[0]
    t = np.linspace(0, n_samples / 500, n_samples)  # fs = 500 Hz
    baseline = amplitude * np.sin(2 * np.pi * freq * t)
    baseline = baseline[:, np.newaxis]  # pour que ça match les 12 leads
    return signal + baseline

def add_powerline_noise(signal, freq=50, snr_db=18):
    """Ajoute un bruit sinusoïdal à 50 Hz (bruit secteur)."""
    n_samples = signal.shape[0]
    t = np.linspace(0, n_samples / 500, n_samples)  # fs = 500 Hz
    sine = np.sin(2 * np.pi * freq * t)
    sine = sine[:, np.newaxis]  # étendre à tous les leads

    # Ajuste l’amplitude du bruit sinusoïdal selon le SNR
    noise_power = calculate_noise_power(signal, snr_db)
    sine_power = np.mean(sine ** 2)
    scaled_sine = sine * np.sqrt(noise_power / sine_power)
    return signal + scaled_sine

def add_motion_artifact(signal, snr_db=12):
    """
    Simule un artefact de mouvement en injectant des impulsions brusques.
    Inspiré de Sörnmo & Laguna, modélisé comme des transitoires de basse fréquence.
    """
    noisy_signal = signal.copy()
    n_samples = signal.shape[0]

    # Une impulsion tous les 1 à 2 secondes (aléatoire)
    num_artifacts = np.random.randint(3, 6)
    positions = np.random.randint(0, n_samples, num_artifacts)

    for pos in positions:
        duration = np.random.randint(100, 300)  # durée en échantillons
        offset = np.random.uniform(-0.4, 0.4)   # amplitude aléatoire
        for lead in range(signal.shape[1]):
            start = pos
            end = min(pos + duration, n_samples)
            noisy_signal[start:end, lead] += offset * np.hanning(end - start)

    # Normalise l’amplitude globale pour respecter le SNR
    noise = noisy_signal - signal
    actual_noise_power = np.mean(noise ** 2)
    target_noise_power = calculate_noise_power(signal, snr_db)
    scaling = np.sqrt(target_noise_power / actual_noise_power)
    return signal + (noise * scaling)

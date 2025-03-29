import os
import numpy as np
import pandas as pd
import wfdb

def load_signals(csv_path, records_path, keep_folds=(1,2,3,4,5,6,7,8,9,10), max_signals=100000):
    print("Étape 1 : Chargement des métadonnées...")
    df = pd.read_csv(csv_path, index_col='ecg_id')
    print(f"→ Total ECG dans CSV : {len(df)}")

    df = df[df.strat_fold.isin(keep_folds)].copy()
    print(f"→ ECG conservés (folds {keep_folds}) : {len(df)}")

    signals = []
    valid_indices = []

    print("Étape 2 : Chargement des signaux à 500 Hz avec WFDB...")
    for i, (idx, filename) in enumerate(df['filename_hr'].items()):
        if i >= max_signals:
            print(f"Arrêt : limite max_signals atteinte ({max_signals})")
            break

        record_path = os.path.join(records_path, filename)
        try:
            signal, _ = wfdb.rdsamp(record_path)
            signals.append(signal)
            valid_indices.append(idx)

            if i % 100 == 0:
                print(f"→ {i} signaux chargés...")

        except Exception as e:
            print(f"Erreur sur {record_path} : {e}")
            continue

    print("Étape 3 : Conversion en array NumPy (float32)...")
    signals = np.array(signals, dtype=np.float32)

    df = df.loc[valid_indices]
    print(f"{len(signals)} signaux valides finalisés.")
    return signals, df



def split_signals(signals, df, val_fold=9, test_fold=10):
    strat_folds = df['strat_fold'].values

    X_train = signals[(strat_folds != val_fold) & (strat_folds != test_fold)]
    X_val   = signals[strat_folds == val_fold]
    X_test  = signals[strat_folds == test_fold]

    return X_train, X_val, X_test

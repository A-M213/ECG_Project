# ECG Denoising with Autoencoder

Ce projet permet d'entraîner un autoencodeur convolutif pour débruiter des signaux ECG issus de la base de données PTB-XL. Le code est structuré pour fonctionner localement avec les données stockées dans un dossier `data/`. Toutes les étapes sont exécutables directement depuis le fichier `main.ipynb`.

---

## Prérequis

- Python ≥ 3.8
- Installation des dépendances :
```bash
pip install -r requirements.txt
```

---

## Préparation des données

1. Télécharger la base de données PTB-XL à l'adresse suivante :  
   https://physionet.org/content/ptb-xl/1.0.3/

2. Télécharger uniquement :
   - Le fichier `ptbxl_database.csv`
   - Le dossier `records500/` contenant les fichiers `.dat` et `.hea` (500 Hz)

3. Placer le tout dans le dossier suivant de votre projet :
```
data/
└── ptbxl/
    ├── ptbxl_database.csv
    └── records500/
        ├── 00000/
        │   ├── 00001_hr.dat
        │   └── 00001_hr.hea
        └── ...
```

> **Attention** : le dossier contenant les données doit impérativement être nommé `ptbxl`.

---

## Structure du projet
```
ECG_Project/
├── data/
│   ├── ptbxl/               # Données brutes téléchargées (voir ci-dessus)
│   ├── processed/           # Signaux propres au format .npy
│   └── noisy/               # Signaux bruités au format .npy
├── models/                  # Modèles sauvegardés (.pth)
├── results/                 # Fichiers .npy et courbes de pertes
├── src/
│   ├── train_autoencoder.py         # Entraînement du modèle
│   ├── eval_autoencoder.py          # Évaluation et métriques
│   ├── generate_noisy_data.py       # Génère des signaux bruités
│   ├── noise.py                     # Fonctions de bruit
│   ├── load_dataset.py              # Chargement WFDB → NumPy
│   ├── dataset.py                   # Chargement initial
│   ├── data/
│   │   └── ECGDataset.py            # Dataset PyTorch
│   └── models/
│       └── autoencoder.py           # Architecture du modèle
├── main.ipynb               # Notebook principal à exécuter
└── README.md
```

---

## Étapes d'exécution

L'ensemble du projet est contrôlé via le fichier `main.ipynb`. Celui-ci contient les cellules suivantes :

1. **Installation des dépendances**
```python
!pip install -r requirements.txt
```

2. **Chargement et sauvegarde des signaux propres**
```python
from src.load_dataset import load_signals, split_signals
signals, df = load_signals(csv_path, records_path)
X_train, X_val, X_test = split_signals(signals, df)
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/X_test.npy", X_test)
```

3. **Visualisation des signaux extraits**
Affichage des 5 premiers signaux d'entraînement (Lead I).

4. **Génération des signaux bruités**
```python
from src.generate_noisy_data import generate_noisy_data
generate_noisy_data()
```

5. **Visualisation des signaux bruités par type de bruit**
Affichage de signaux propres et bruités pour chaque type de bruit artificiel.

6. **Entraînement de l’autoencodeur**
```python
from src.train_autoencoder import train_autoencoder
train_autoencoder(batch_size=32, epochs=20, lr=1e-3)
```
Les courbes de loss sont sauvegardées dans `results/train_losses.npy` et `results/val_losses.npy`.

7. **Affichage de la courbe de perte (loss)**
Affichage de l'évolution de la MSE au fil des époques.

8. **Évaluation du modèle sur les signaux test**
```python
from src.eval_autoencoder import evaluate_autoencoder

evaluate_autoencoder(
    model_path = "models/autoencoder.pth",
    path_noisy = "data/noisy/X_test_noisy.npy",
    path_clean = "data/processed/X_test.npy",
    path_labels = "data/noisy/X_test_noisy_labels.npy"
)
```
Cette étape affiche les métriques (MSE, SNR, Corrélation) globales et par type de bruit.

9. **Affichage comparatif des signaux propres, bruités et débruités**
Pour chaque type de bruit : affichage côte à côte du signal propre, bruité, et débruité (Lead I).

---

## Auteur
Projet réalisé dans le cadre d'un test de compétence IA, 2025.


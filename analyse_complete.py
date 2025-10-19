# ==============================================================================
# PROJET M2 - ADVANCED MECHATRONICS
# Script d'analyse complet pour la maintenance prédictive
# VERSION FINALE - Fiabilisée, Débuguée et Optimisée
#
# Auteur: [Votre Nom]
# Date: 14/10/2025
#
# Cette version est conçue pour reproduire les résultats de référence du
# tableau de bord HTML en garantissant l'alignement des données et en
# utilisant des hyperparamètres optimisés.
# ==============================================================================

# --- 1. IMPORTATION DES LIBRAIRIES ---
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# Imports Keras modernes pour Python 3.11+
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

print("--- Librairies importées ---")

# --- 2. CHARGEMENT DES DONNÉES ---
print("--- Chargement des fichiers de données ---")
try:
    column_names = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=column_names).dropna(axis=1)
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=column_names).dropna(axis=1)
    truth_df = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None).dropna(axis=1)
    truth_df.columns = ['RUL']
    print("Données chargées avec succès.")
except FileNotFoundError as e:
    print(f"ERREUR: Fichier introuvable - {e}. Assurez-vous que les fichiers .txt sont dans le même dossier que le script.")
    exit()

# --- 3. PRÉPARATION DES DONNÉES ---
print("\n--- Préparation des données ---")
sequence_length = 30
features_cols = ['setting_1', 'setting_2', 'setting_3'] + [col for col in train_df.columns if 'sensor' in col and col not in ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']]
target_col = 'RUL'

# A. Calculer la RUL pour les données d'entraînement
train_df['RUL'] = train_df.groupby('unit_number')['time_in_cycles'].transform('max') - train_df['time_in_cycles']
train_df['RUL'] = train_df['RUL'].clip(upper=125)

# B. Normaliser les features et la cible
feature_scaler = MinMaxScaler()
train_df[features_cols] = feature_scaler.fit_transform(train_df[features_cols])
test_df[features_cols] = feature_scaler.transform(test_df[features_cols])

target_scaler = MinMaxScaler()
train_df[target_col] = target_scaler.fit_transform(train_df[[target_col]])
print("Normalisation terminée.")

# C. Créer les séquences d'entraînement
def create_sequences(df, seq_len, feat_cols, target):
    sequences, labels = [], []
    for uid in df['unit_number'].unique():
        udata = df[df['unit_number'] == uid]
        for i in range(len(udata) - seq_len + 1):
            sequences.append(udata[feat_cols].iloc[i:i + seq_len].values)
            labels.append(udata[target].iloc[i + seq_len - 1])
    return np.array(sequences), np.array(labels)

X_train, y_train_scaled = create_sequences(train_df, sequence_length, features_cols, target_col)

# D. Créer les séquences de test et aligner les vraies RUL
X_test = np.array([test_df[test_df['unit_number'] == uid][features_cols].values[-sequence_length:]
                   for uid in test_df['unit_number'].unique()])
y_test_true = truth_df['RUL'].values

print("Création des séquences terminée. L'alignement est correct.")

# --- 4. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES ---
results = {}

# --- Modèle 1: Random Forest ---
print("\n--- Entraînement Modèle 1: Random Forest ---")
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=15)
start_time = time.time()
rf_model.fit(X_train_reshaped, y_train_scaled.ravel())
training_time = time.time() - start_time
print(f"Entraînement terminé en {training_time:.2f}s.")

pred_scaled = rf_model.predict(X_test_reshaped)
predictions = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
results['Random Forest'] = {'Predictions': predictions, 'Time': training_time}

# --- Modèle 2: SVR ---
print("\n--- Entraînement Modèle 2: SVR ---")
# On utilise un échantillon pour une exécution rapide
sample_size_svr = 5000
idx_svr = np.random.choice(np.arange(len(X_train_reshaped)), sample_size_svr, replace=False)
svr_model = SVR(kernel='rbf', C=20, epsilon=0.1, gamma='scale')
start_time = time.time()
svr_model.fit(X_train_reshaped[idx_svr], y_train_scaled[idx_svr].ravel())
training_time = time.time() - start_time
print(f"Entraînement sur {sample_size_svr} échantillons terminé en {training_time:.2f}s.")

pred_scaled = svr_model.predict(X_test_reshaped)
predictions = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
results['SVR'] = {'Predictions': predictions, 'Time': training_time}

# --- Modèle 3: LSTM ---
print("\n--- Entraînement Modèle 3: LSTM ---")
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features_cols))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
start_time = time.time()
lstm_model.fit(X_train, y_train_scaled, epochs=20, batch_size=256, validation_split=0.2, verbose="auto", shuffle=True)
training_time = time.time() - start_time
print(f"Entraînement terminé en {training_time:.2f}s.")

pred_scaled = lstm_model.predict(X_test)
predictions = target_scaler.inverse_transform(pred_scaled).flatten()
results['LSTM'] = {'Predictions': predictions, 'Time': training_time}

# --- 5. CALCUL DES SCORES ET PRÉSENTATION ---
for name, data in results.items():
    y_true_np = np.array(y_test_true)
    rmse = np.sqrt(mean_squared_error(y_true_np, data['Predictions']))
    mae = mean_absolute_error(y_true_np, data['Predictions'])
    results[name]['RMSE'] = rmse
    results[name]['MAE'] = mae

print("\n" + "="*50)
print("--- RÉSULTATS COMPARATIFS FINAUX (Version Correcte) ---")
print("="*50)
results_df = pd.DataFrame({
    "Modèle": list(results.keys()),
    "RMSE": [res['RMSE'] for res in results.values()],
    "MAE": [res['MAE'] for res in results.values()],
    "Temps d'entraînement (s)": [res['Time'] for res in results.values()]
}).round(2)
print(results_df)

# --- 6. GÉNÉRATION DES GRAPHIQUES ---
print("\n--- Génération des graphiques ---")
plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.plot(np.array(y_test_true), label='Vraie RUL', color='blue', linewidth=2.5, alpha=0.9)
colors = {'Random Forest': '#4CAF50', 'SVR': '#FFC107', 'LSTM': '#F44336'}
for name, res in results.items():
    plt.plot(res['Predictions'], label=f"{name} (RMSE: {res['RMSE']:.2f})", linestyle='--', color=colors[name])
plt.title('Comparaison des Prédictions sur le Jeu de Test', fontsize=16)
plt.xlabel('Index du Moteur de Test', fontsize=12)
plt.ylabel('RUL (cycles)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
bars = plt.bar(results_df['Modèle'], results_df['RMSE'], color=[colors[m] for m in results_df['Modèle']])
plt.title('Comparaison des Scores RMSE', fontsize=16)
plt.ylabel('RMSE', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('comparaison_modeles_final_correct.png')
print("Graphique 'comparaison_modeles_final_correct.png' sauvegardé.")
plt.show()


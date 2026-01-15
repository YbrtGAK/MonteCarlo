# Analyse d'incertitude par Monte Carlo sur des mesures en régime permanent
# Yann Berton - Ebullition convective

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# 1. Préparation des données (exemple)
# Supposons que vous ayez un DataFrame avec les colonnes T1, P1, m_dot
np.random.seed(42)  # Pour la reproductibilité
data = {
    'T1': np.random.normal(273.15, 0.5, 30),  # 30 points pour l'exemple
    'P1': np.random.normal(1e5, 500, 30),
    'm_dot': np.random.normal(0.1, 0.01, 30)
}
df = pd.DataFrame(data)

# Affichage des données
print("Aperçu des données :")
print(df.head())

# 2. Calcul des moyennes et incertitudes-types sur les moyennes
moyennes = df.mean()
n = len(df)
incertitudes_moyennes = df.std() / np.sqrt(n)
incertitudes_elargies = 2 * incertitudes_moyennes

print("\nMoyennes :\n", moyennes)
print("\nIncertitudes-types sur les moyennes :\n", incertitudes_moyennes)
print("\nIncertitudes élargies (k=2) :\n", incertitudes_elargies)

# 3. Vérification de la stationnarité (optionnel)
df.plot(subplots=True, layout=(3,1), figsize=(10, 8))
plt.suptitle("Séries temporelles des mesures")
plt.show()

# 4. Simulation Monte Carlo sur les moyennes
n_simulations = 10**6

# Génération des échantillons aléatoires
T1_samples = np.random.normal(moyennes['T1'], incertitudes_moyennes['T1'], n_simulations)
P1_samples = np.random.normal(moyennes['P1'], incertitudes_moyennes['P1'], n_simulations)
m_dot_samples = np.random.normal(moyennes['m_dot'], incertitudes_moyennes['m_dot'], n_simulations)

# Calcul de l'enthalpie (exemple pour du R134a)
h_samples = np.array([CP.PropsSI('H', 'T', T, 'P', P, 'R134a') for T, P in zip(T1_samples, P1_samples)])

# Analyse des résultats
h_mean = np.mean(h_samples)
h_std = np.std(h_samples)
incertitude_elargie_h = 2 * h_std

print(f"\nEnthalpie moyenne : {h_mean:.2f} J/kg")
print(f"Incertitude élargie (k=2) : {incertitude_elargie_h:.2f} J/kg")

# 5. Visualisation de la distribution de l'enthalpie
plt.hist(h_samples, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(h_mean, color='k', linestyle='dashed', linewidth=1)
plt.axvline(h_mean + incertitude_elargie_h, color='r', linestyle='dotted', linewidth=1)
plt.axvline(h_mean - incertitude_elargie_h, color='r', linestyle='dotted', linewidth=1)
plt.title("Distribution de l'enthalpie (Monte Carlo)")
plt.xlabel("Enthalpie (J/kg)")
plt.ylabel("Densité de probabilité")
plt.show()

# 6. Prise en compte des incertitudes systématiques (exemple)
# Incertitude systématique uniforme de ±0.5 K sur T1
T1_sys = np.random.uniform(-0.5, 0.5, n_simulations)
T1_samples_sys = T1_samples + T1_sys

# Recalcul de l'enthalpie avec incertitude systématique
h_samples_sys = np.array([CP.PropsSI('H', 'T', T, 'P', P, 'R134a') for T, P in zip(T1_samples_sys, P1_samples)])

h_mean_sys = np.mean(h_samples_sys)
h_std_sys = np.std(h_samples_sys)
incertitude_elargie_h_sys = 2 * h_std_sys

print(f"\nEnthalpie moyenne (avec incertitude systématique) : {h_mean_sys:.2f} J/kg")
print(f"Incertitude élargie (k=2) : {incertitude_elargie_h_sys:.2f} J/kg")

# 7. Prise en compte des corrélations (exemple)
# Matrice de covariance pour T1 et P1
cov_matrix = np.array([[incertitudes_moyennes['T1']**2, 0.5*incertitudes_moyennes['T1']*incertitudes_moyennes['P1']],
                       [0.5*incertitudes_moyennes['T1']*incertitudes_moyennes['P1'], incertitudes_moyennes['P1']**2]])
T1_P1_samples = np.random.multivariate_normal([moyennes['T1'], moyennes['P1']], cov_matrix, n_simulations)
T1_samples_corr, P1_samples_corr = T1_P1_samples[:, 0], T1_P1_samples[:, 1]

# Calcul de l'enthalpie avec corrélations
h_samples_corr = np.array([CP.PropsSI('H', 'T', T, 'P', P, 'R134a') for T, P in zip(T1_samples_corr, P1_samples_corr)])
h_mean_corr = np.mean(h_samples_corr)
h_std_corr = np.std(h_samples_corr)
incertitude_elargie_h_corr = 2 * h_std_corr
print(f"\nEnthalpie moyenne (avec corrélations) : {h_mean_corr:.2f} J/kg")
print(f"Incertitude élargie (k=2) : {incertitude_elargie_h_corr:.2f} J/kg")

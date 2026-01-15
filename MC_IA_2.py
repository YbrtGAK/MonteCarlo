# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:48:04 2025

@author: yberton
"""

import numpy as np
import matplotlib.pyplot as plt

# === PARAMÈTRES DE SIMULATION ===
N_MONTE_CARLO = 100000  # Nombre de tirages
temp_consigne = np.array([15, 30, 45, 60, 75, 90, 105, 120])  # °C

# Exemple : mesures d’un thermocouple (à remplacer par les vraies données)
# Chaque élément est une mesure moyenne obtenue à chaque palier de consigne
mesures_thermocouple = np.array([15.039945482352937, 29.965301178888886, 44.92675827129629, 59.86238288928571, 74.74012391304346, 89.92970367599999, 104.95061072413792, 120.08327836363637])  # °C

# === ÉTAPE 1 : Ajustement de la loi affine ===
coeffs, cov = np.polyfit(mesures_thermocouple, temp_consigne, deg=1, cov=True)
a, b = coeffs
u_a, u_b = np.sqrt(np.diag(cov))  # incertitudes-type sur a et b

# Estimation des résidus (erreurs d’ajustement)
residus = temp_consigne - (a * mesures_thermocouple + b)
ecartype_residus = np.std(residus, ddof=1)

# === ÉTAPE 2 : Simulation Monte Carlo ===
def monte_carlo_uncertainty(x_mesure):
    """
    Calcule l'incertitude sur la température calibrée à partir d'une mesure x_mesure
    """
    # Tirages aléatoires des paramètres a et b selon leurs incertitudes
    a_rand = np.random.normal(loc=a, scale=u_a, size=N_MONTE_CARLO)
    b_rand = np.random.normal(loc=b, scale=u_b, size=N_MONTE_CARLO)

    # Modèle linéaire simulé : T = a_rand * x + b_rand
    bruit_rand = np.random.normal(loc=0, scale=ecartype_residus, size=N_MONTE_CARLO)
    T_simul = a_rand * x_mesure + b_rand + bruit_rand

    # Moyenne et incertitude-type de la distribution simulée
    moyenne_T = np.mean(T_simul)
    u_T = np.std(T_simul, ddof=1)

    return moyenne_T, u_T, T_simul



# === Exemple d'application ===
x_nouvelle = 82.0  # température mesurée par le thermocouple (brute)
T_estimee, incertitude, distribution = monte_carlo_uncertainty(x_nouvelle)

def incertitude_temperature(x, a, b, cov_ab, sigma_mod):
    """
    Calcule l'incertitude-type u_T pour une mesure brute x
    - a, b : coefficients de la régression
    - cov_ab : matrice 2x2 de covariance de [a, b]
    - sigma_mod : écart-type des résidus (modélisation)
    """
    var_a, cov_ab_, var_b = cov_ab[0,0], cov_ab[0,1], cov_ab[1,1]
    u2 = x**2 * var_a + var_b + 2 * x * cov_ab_ + sigma_mod**2
    return np.sqrt(u2)


print(f"Mesure brute : {x_nouvelle} °C")
print(f"Température estimée calibrée : {T_estimee:.2f} ± {incertitude:.2f} °C (k=1)")

# === Histogramme de la distribution simulée ===
plt.hist(distribution, bins=100, density=True, alpha=0.7, color='skyblue')
plt.axvline(T_estimee, color='red', linestyle='--', label='Moyenne simulée')
plt.title("Distribution Monte Carlo de la température corrigée")
plt.xlabel("Température (°C)")
plt.ylabel("Densité")
plt.legend()
plt.grid(True)
plt.show()

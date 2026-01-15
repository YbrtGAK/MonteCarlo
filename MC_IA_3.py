# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:07:50 2025

@author: yberton
"""

import numpy as np
import CoolProp.CoolProp as CP

# Constantesuv add
N = 10000  # Nombre d'échantillons Monte Carlo
fluide = 'Water'  # à adapter selon ton fluide

# Mesures (exemple) et incertitudes (1σ)
mdot_nom = 0.5        # kg/s
u_mdot = 0.01         # kg/s

Tin_nom = 70.0        # °C
u_Tin = 0.2           # °C

Tout_nom = 90.0       # °C
u_Tout = 0.2          # °C

P_nom = 2.0e5         # Pa
u_P = 500.0           # Pa

# === Échantillonnage aléatoire des grandeurs mesurées ===
mdot_samples = np.random.normal(mdot_nom, u_mdot, N)
Tin_samples = np.random.normal(Tin_nom, u_Tin, N) + 273.15  # °C → K
Tout_samples = np.random.normal(Tout_nom, u_Tout, N) + 273.15
P_samples = np.random.normal(P_nom, u_P, N)

# === Calcul de cp (CoolProp) en chaque point ===
cp_in = np.array([CP.PropsSI('C', 'T', T, 'P', P, fluide) for T, P in zip(Tin_samples, P_samples)])
cp_out = np.array([CP.PropsSI('C', 'T', T, 'P', P, fluide) for T, P in zip(Tout_samples, P_samples)])
cp_mean = 0.5 * (cp_in + cp_out)  # J/kg/K

# === Calcul de la puissance thermique ===
delta_T = Tout_samples - Tin_samples  # en K
P_samples = mdot_samples * cp_mean * delta_T  # en W

# === Résultat Monte Carlo ===
P_moy = np.mean(P_samples)
u_P = np.std(P_samples, ddof=1)
k = 1.96  # pour 95 % si normal

print(f"Puissance moyenne : {P_moy:.2f} W")
print(f"Incertitude-type : {u_P:.2f} W")
print(f"Intervalle de confiance 95 % : ±{k*u_P:.2f} W")

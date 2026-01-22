import pandas as pd
import numpy as np
from CoolProp.CoolProp import PropsSI

# Définition de la fonction H
H = lambda P, T: PropsSI('H', 'P', P*1E5, 'T', T+273.15, 'R245fa')

# Exemple de data
#####BASE DE DONNEES INDEPENDANTE ETALONNAGE#####
df = lvm_to_df(r".\exemples\PPh_359.lvm")
df.drop(columns=["Comment"],inplace=True)

# Vectorisation de la fonction H
H_vectorized = np.vectorize(H)

# Application de H_vectorized sur chaque élément des listes de A et B
df['C'] = [H_vectorized(a, b).tolist() for a, b in zip(df['A'], df['B'])]

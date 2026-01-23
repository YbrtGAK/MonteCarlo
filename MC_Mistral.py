import pandas as pd
import numpy as np
from CoolProp.CoolProp import PropsSI
from utilities.data.lvm import lvm_to_df

# Définition de la fonction H
H = lambda P, T: PropsSI('H', 'P', P*1E5, 'T', T+273.15, 'R245fa')

# Exemple de data
#####BASE DE DONNEES INDEPENDANTE ETALONNAGE#####
df = lvm_to_df(r".\exemples\PPh_359.lvm")
df.drop(columns=["Comment"],inplace=True)

# Vectorisation de la fonction H
H_vectorized = np.vectorize(H)

# Application de H_vectorized sur chaque élément des listes de A et B
df['H'] = [H_vectorized(a, b).tolist() for a, b in zip(df['118 - P_TS_in [bars]'], df['202 - E_in_imm [°C]'])]
print(f"df['H'] = {df['H']}")
import numpy as np
import matplotlib.pyplot as plt

class ThermocoupleEtalonne:
    def __init__(self, x_raw, T_ref):
        """
        Initialise l'objet avec :
        - x_raw : valeurs mesurées par le thermocouple (moyennes brutes)
        - T_ref : températures de consigne correspondantes
        """
        self.x_raw = np.array(x_raw)
        self.T_ref = np.array(T_ref)
        assert len(self.x_raw) == len(self.T_ref), "Les vecteurs doivent être de même longueur"
        
        self._fit()
    
    def _fit(self):
        """
        Régression linéaire avec estimation des incertitudes.
        """
        X = np.vstack([self.x_raw, np.ones(len(self.x_raw))]).T  # matrice [x, 1]
        Y = self.T_ref
        
        # Régression linéaire : moindres carrés
        beta, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        self.a, self.b = beta

        # Estimation de l’incertitude sur a et b
        n = len(Y)
        y_pred = self.a * self.x_raw + self.b
        res = Y - y_pred
        s2 = np.sum(res**2) / (n - 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_beta = s2 * XtX_inv
        
        self.cov = cov_beta
        self.residual_std = np.std(res, ddof=2)  # sigma_mod

    def corriger(self, x):
        """
        Applique la correction sur une ou plusieurs mesures x.
        Retourne :
        - Température corrigée
        - Incertitude-type associée (float ou array selon x)
        """
        x = np.atleast_1d(x)
        T_corr = self.a * x + self.b

        var_a = self.cov[0,0]
        var_b = self.cov[1,1]
        cov_ab = self.cov[0,1]
        
        u_T2 = x**2 * var_a + var_b + 2 * x * cov_ab + self.residual_std**2
        u_T = np.sqrt(u_T2)

        if u_T.size == 1:
            return T_corr[0], u_T[0]
        return T_corr, u_T

    def rapport(self, with_plot=True):
        """
        Affiche les résultats de l'étalonnage et un graphique de validation.
        """
        print("=== Résultats de l'étalonnage ===")
        print(f"Loi : T = {self.a:.5f} * x + {self.b:.5f}")
        print(f"Incertitude-type sur a : {np.sqrt(self.cov[0,0]):.5f}")
        print(f"Incertitude-type sur b : {np.sqrt(self.cov[1,1]):.5f}")
        print(f"Covariance a,b : {self.cov[0,1]:.5e}")
        print(f"Erreur de modélisation (écart-type des résidus) : {self.residual_std:.5f} °C")
        
        if with_plot:
            x_pred = np.linspace(min(self.x_raw), max(self.x_raw), 100)
            T_pred, u_pred = self.corriger(x_pred)
            
            plt.figure(figsize=(8, 5))
            plt.errorbar(self.x_raw, self.T_ref, yerr=self.residual_std, fmt='o', label="Données étalonnage")
            plt.plot(x_pred, T_pred, label="Loi affine", color="red")
            plt.fill_between(x_pred, T_pred - u_pred, T_pred + u_pred, color="red", alpha=0.3, label="±1σ")
            plt.xlabel("Mesure brute du thermocouple")
            plt.ylabel("Température (°C)")
            plt.legend()
            plt.grid(True)
            plt.title("Étalonnage du thermocouple")
            plt.tight_layout()
            plt.show()

# Mesures moyennes sur 8 paliers
x_mesures = np.array([15.039945482352937, 29.965301178888886, 44.92675827129629, 59.86238288928571, 74.74012391304346, 89.92970367599999, 104.95061072413792, 120.08327836363637])  # °C
T_consigne = [15, 30, 45, 60, 75, 90, 105, 120]  # température de référence

# Création de l’objet
tc = ThermocoupleEtalonne(x_mesures, T_consigne)
tc.rapport()

# Utilisation sur une nouvelle mesure brute :
x_nouveau = 55.32
T_corr, u_T = tc.corriger(x_nouveau)
print(f"Mesure corrigée : {T_corr:.2f} °C ± {u_T:.2f} °C (1σ)")

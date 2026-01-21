
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


                                        ProbabilisticDataFrame

Class for automatic uncertainity propagation for dataframe calculation.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Imports
import pandas as pd
from scipy.stats import chi2 
import numpy as np

np.random.seed(42) # Seed fixed for random sample generations

# Class definition
class ProbabilisticDataFrame():

    """Class for automatic uncertainity propagation for dataframe calculation."""

    def __init__(self, df:pd.Series|pd.DataFrame, udf:pd.Series|pd.DataFrame) :

        self.df = df
        self.udf = udf
        self.check()
 
    def check(self) -> None :

        """Check if the table of values and the table of uncertainties have the same shape
        and the same name of columns"""

        if self.df.shape != self.udf.shape :
            raise ValueError(f"""
        Dataframes don't have the same dimensions : 
           - df.shape = {df.shape} 
           - udf.shape = {udf.shape}
                            """)
        
        if False in [df.columns[i] == udf.columns[i] for i in range(len(df.columns))] : 
            raise ValueError(f"""Dataframes don't have the same column names :
                df.columns : {df.columns}
                udf.columns : {udf.columns}""")
        
    def propagate(self, Xk:list[str], exp:list[str], alpha : float) -> None :

        """Allow the user to calculate the expression exp and the uncertainty over
        the output value with Monte Carlo Simulation"""

        if len([l for l in exp[-1].split('X')[1:] if l[0].isdigit()]) != len(Xk) :
            raise ValueError("""
            Xk and exp have different variable numbers.
            If not, make sure there is no Xi with i a number 
            that is not a variable in your expression.""")
        self.exp = exp #### A RETIRER
        N = self.N_samples_calculation(alpha) # Determination of the number of simulation samples N
        nVar = len(Xk)
        exp_df = exp_udf = exp_df_mc = exp[-1] # Initialize expressions for df and udf

        df_mc = pd.DataFrame(index=range(len(self.df)), columns=Xk)
        # Replace the variables by their name in both expressions
        for i in range(nVar) :
            exp_df = exp_df.replace('X'+str(i), "self.df['" + Xk[i] + "']")
            exp_udf = exp_udf.replace('X'+str(i), "self.udf['" + Xk[i] + "']")
            exp_df_mc = exp_df_mc.replace('X'+str(i), "self.df_mc['" + Xk[i] + "']")

        ## Generate random samples for each Xk and calculate their mean and std
            samples = np.random.normal(loc = self.df[Xk[i]].values,
                                       scale = self.udf[Xk[i]].values,
                                       size = (N,len(self.df)))
            list_samples = [pd.Series(samples[:,index]) for index in range(len(samples[0]))]
            df_mc[Xk[i]] = list_samples

        self.df[exp[0]] = eval(exp_df) # Evaluate the expression for df
        
        # Monte Carlo simulations for uncertainty calculation

        ## Random population initialization for the variables Xk
        self.df_mc = self.random_population_initialization(Xk,N)

        ## Evaluate the expression with df_mc
        self.df_mc[exp[0]] = eval(exp_df_mc)
        self.df[exp[0]] = self.df_mc[exp[0]].apply(np.mean)
        self.udf[exp[0]] = self.df_mc[exp[0]].apply(np.std)
        
    def N_samples_calculation(self, alpha:float) -> int :
        """Determination of the number of simulation samples N"""
        ## Calculation of the interval [a <= s(Xk)/σ(Xk) <= b] bounds
        a = (1-alpha) # a -> lower bound
        b = 2 - a # b -> upper bound
        delta = b**2 - a**2
        ## Calculation by iteration of the number of samples N - Condition on $\chi²$
        nu = 1 # degree of freedom ν = N - 1
        diff = (chi2.ppf((1-alpha/2),nu) - chi2.ppf(alpha/2,nu))/nu
        while (0.9999 <= diff/delta <= 1.0001) == False:
            nu += 1
            diff = (chi2.ppf((1-alpha/2),nu) - chi2.ppf(alpha/2,nu))/nu
        N = (nu + 1)*2 # Overestimation of N to make sure of the independance of the samples
        return(N)

    def random_population_initialization(self, Xk : list[str], N : int) -> pd.DataFrame : 

        """Generate for each Xk variable a normal distribution over its mean value.
        Xk can be a scalar or an array (pd.Dataframe or pd.Series)"""

        ## Initialization of an empty dataframe which shape == df.shape
        df_mc = pd.DataFrame(index=range(len(df)), columns=Xk)
        ## Generate random samples for each Xk and calculate their mean and std
        for X in Xk :
            samples = np.random.normal(loc = self.df[X].values,
                                       scale = self.udf[X].values,
                                       size = (N,len(self.df)))
            list_samples = [pd.Series(samples[:,index]) for index in range(len(samples[0]))]
            df_mc[X] = list_samples
        return(df_mc)

if __name__ == "__main__" :

    """Quick test to use the probilistic DataFrame"""
    data = {'A' : [1,2], 'B' : [3,4]}
    udata = {'A' : [0.01,0.02], 'B' : [0.03,0.04]}
    df = pd.DataFrame(data)
    udf = pd.DataFrame(udata)
    pdf = ProbabilisticDataFrame(df,udf)
    pdf.propagate(['A','B'],exp=["C","X0*X1"],alpha=0.05)

        
    
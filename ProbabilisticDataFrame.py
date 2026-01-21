
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                        ProbabilisticDataFrame

Class for automatic uncertainity propagation for dataframe calculation.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Imports
import pandas as pd
from scipy.stats import chi2 
import numpy as np
from CoolProp.CoolProp import PropsSI

H = lambda P,T : PropsSI('H', 'P', P*1E5, 'T', T+273.15, 'R245fa')
np.random.seed(42) # Seed fixed for random sample generations

# Class definition
class ProbabilisticDataFrame():

    """Class for automatic uncertainity propagation for dataframe calculation."""

    def __init__(self, df:pd.Series|pd.DataFrame, udf:pd.Series|pd.DataFrame,
                 **kwargs) :

        self.df = df
        self.udf = udf
        self.check()
 
    def check(self) -> None :

        """Check if the table of values and the table of uncertainties have the same shape
        and the same name of columns"""

        if self.df.shape != self.udf.shape :
            raise ValueError(f"""
        Dataframes don't have the same dimensions : 
           - df.shape = {self.df.shape} 
           - udf.shape = {self.udf.shape}
                            """)
        if type(self.df) == pd.core.frame.DataFrame : 
            if False in [self.df.columns[i] == self.udf.columns[i] for i in range(len(self.df.columns))] : 
                raise ValueError(f"""Dataframes don't have the same column names :
                    df.columns : {self.df.columns}
                    udf.columns : {self.udf.columns}""")
        elif pd.core.series.Series :
            if False in [self.df.index[i] == self.udf.index[i] for i in range(len(self.df.index))] : 
                raise ValueError(f"""Series don't have the same column names :
                    df.index : {self.df.index}
                    udf.index : {self.udf.index}""")
        
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

        if type(self.df) == pd.core.frame.DataFrame : 
            self.df_mc = pd.DataFrame(index=range(len(self.df)), columns=Xk)
            for i in range(nVar) :
                exp_df = exp_df.replace('X'+str(i), "self.df['" + Xk[i] + "'].values")
                exp_udf = exp_udf.replace('X'+str(i), "self.udf['" + Xk[i] + "'].values")
                exp_df_mc = exp_df_mc.replace('X'+str(i), "self.df_mc['" + Xk[i] + "'].values")

                samples = np.random.normal(loc = self.df[Xk[i]].values,
                                        scale = self.udf[Xk[i]].values,
                                        size = (N,len(self.df)))
                
                list_samples = [pd.Series(samples[:,index]) for index in range(len(samples[0]))]

                self.df_mc[Xk[i]] = [list_samples]
                    ## Generate random samples for each Xk and calculate their mean and std
  
            # Monte Carlo simulations for uncertainty calculation

            ## Random population initialization for the variables Xk

            ## Evaluate the expression with df_mc
            self.df_mc[exp[0]] = eval(exp_df_mc)
            self.df[exp[0]] = self.df_mc[exp[0]].apply(np.mean)
            self.udf[exp[0]] = self.df_mc[exp[0]].apply(np.std)

        elif pd.core.series.Series :
            self.df_mc = pd.DataFrame(index=range(N), columns=Xk)
            for i in range(nVar) :
                exp_df = exp_df.replace('X'+str(i), "self.df['" + Xk[i] + "'].values")
                exp_udf = exp_udf.replace('X'+str(i), "self.udf['" + Xk[i] + "'].values")
                exp_df_mc = exp_df_mc.replace('X'+str(i), "self.df_mc['" + Xk[i] + "'].values")

                list_samples = np.random.normal(loc=self.df[Xk[i]], scale=self.udf[Xk[i]], size=N)
                self.df_mc[Xk[i]] = list_samples  # Chaque colonne est une Series

                    ## Generate random samples for each Xk and calculate their mean and std
            # Monte Carlo simulations for uncertainty calculation
            print(f'exp_df_mc : {exp_df_mc}')
            ## Random population initialization for the variables Xk

            ## Evaluate the expression with df_mc
            self.df_mc[exp[0]] = eval(exp_df_mc)
            self.df[exp[0]] = self.df_mc[exp[0]].mean()
            self.udf[exp[0]] = self.df_mc[exp[0]].std(ddof=1)

        
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
    
    def add_functions(self,**kwargs):
        print(type(**kwargs))

if __name__ == "__main__" :

    """Quick test to use the probilistic DataFrame"""
    """
    data = {'A' : [1,2], 'B' : [3,4]}
    udata = {'A' : [0.01,0.02], 'B' : [0.03,0.04]}
    df = pd.DataFrame(data)
    udf = pd.DataFrame(udata)
    pdf = ProbabilisticDataFrame(df,udf)
    pdf.propagate(['A','B'],exp=["C","X0*X1"],alpha=0.05)
"""
    """Quick test to use the probilistic Serie"""       
    data = {'A' : [1,2], 'B' : [3,4]}
    udata = {'A' : [0.01,0.02], 'B' : [0.03,0.04]}
    df = pd.DataFrame(data).mean()
    udf = pd.DataFrame(udata).std()
    pdf = ProbabilisticDataFrame(df,udf)
    pdf.propagate(['A','B'],exp=["C","X0*X1"],alpha=0.05)
        
    
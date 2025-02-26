import numpy as np
import pandas as pd


class Results:
    """Class to hold estimation results."""
    
    def __init__(self, model, params, param_names, loglikelihood, residuals, 
                 std_residuals, conditional_variance, cov_params=None):
        """Initialize Results object.
        
        Parameters
        ----------
        model : UnivariateTimeSeries
            The model that was estimated
        params : np.ndarray
            Estimated parameters
        param_names : list
            Names of parameters
        loglikelihood : float
            Log-likelihood at the optimum
        residuals : np.ndarray
            Model residuals
        std_residuals : np.ndarray
            Standardized residuals
        conditional_variance : np.ndarray
            Conditional variances
        cov_params : np.ndarray, optional
            Covariance matrix of parameters
        """
        self.model = model
        self.params = params
        self.param_names = param_names
        self.loglikelihood = loglikelihood
        self.residuals = residuals
        self.std_residuals = std_residuals
        self.conditional_variance = conditional_variance
        self.cov_params = cov_params
        
        # Compute information criteria
        self.nobs = len(residuals)
        self.k = len(params)
        self.aic = -2 * loglikelihood + 2 * self.k
        self.bic = -2 * loglikelihood + np.log(self.nobs) * self.k
    
    def summary(self):
        """Print a summary of the estimation results."""
        params_df = pd.DataFrame({
            'Parameter': self.param_names,
            'Estimate': self.params
        })
        
        if self.cov_params is not None:
            std_errors = np.sqrt(np.diag(self.cov_params))
            t_values = self.params / std_errors
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_values)))
            
            params_df['Std. Error'] = std_errors
            params_df['t-value'] = t_values
            params_df['p-value'] = p_values
        
        print("Model Estimation Results")
        print("========================")
        print(f"Log-likelihood: {self.loglikelihood:.4f}")
        print(f"AIC: {self.aic:.4f}")
        print(f"BIC: {self.bic:.4f}")
        print(f"Number of observations: {self.nobs}")
        print("\nParameters:")
        print(params_df.to_string(index=False)) 
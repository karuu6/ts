import numpy as np
import pandas as pd
from typing import List, Any


class Results:
    """Class to hold estimation results."""

    def __init__(self, model: Any, params: np.ndarray, param_names: List[str],
                 loglikelihood: float, residuals: np.ndarray,
                 std_residuals: np.ndarray,
                 ) -> None:
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

        # Compute information criteria
        self.nobs = len(residuals)
        self.k = len(params)
        self.aic = -2 * loglikelihood + 2 * self.k
        self.bic = -2 * loglikelihood + np.log(self.nobs) * self.k

    def summary(self) -> None:
        """Print a summary of the estimation results."""
        params_df = pd.DataFrame({
            'Parameter': self.param_names,
            'Estimate': self.params
        })

        print("Model Estimation Results")
        print("========================")
        print(f"Log-likelihood: {self.loglikelihood:.4f}")
        print(f"AIC: {self.aic:.4f}")
        print(f"BIC: {self.bic:.4f}")
        print(f"Number of observations: {self.nobs}")
        print("\nParameters:")
        print(params_df.to_string(index=False))

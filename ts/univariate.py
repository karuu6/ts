import numpy as np
import pandas as pd
from scipy import optimize
from typing import List, Optional, Tuple, Dict, Any, Union, Callable

from .mean import MeanModel, Constant as ConstantMean
from .variance import VarianceModel, Constant as ConstantVariance
from .distributions import Distribution, Normal
from .results import Results


class UnivariateTimeSeries:
    """Univariate time series model with flexible mean, variance, and distribution specifications."""
    
    def __init__(self, mean_model: Optional[MeanModel] = None, 
                 variance_model: Optional[VarianceModel] = None, 
                 distribution: Optional[Distribution] = None) -> None:
        """Initialize the univariate time series model.
        
        Parameters
        ----------
        mean_model : MeanModel, optional
            Model for the conditional mean. Default is Constant.
        variance_model : VarianceModel, optional
            Model for the conditional variance. Default is Constant.
        distribution : Distribution, optional
            Distribution for the innovations. Default is Normal.
        """
        self.mean_model = mean_model if mean_model is not None else ConstantMean()
        self.variance_model = variance_model if variance_model is not None else ConstantVariance()
        self.distribution = distribution if distribution is not None else Normal()
        
        # Store data
        self.data: Optional[np.ndarray] = None
        self.nobs: int = 0
    
    def _split_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split parameters into mean, variance, and distribution parameters.
        
        Parameters
        ----------
        params : np.ndarray
            Full parameter vector
            
        Returns
        -------
        tuple
            (mean_params, variance_params, dist_params)
        """
        n_mean = self.mean_model.num_params()
        n_var = self.variance_model.num_params()
        
        mean_params = params[:n_mean]
        variance_params = params[n_mean:n_mean+n_var]
        dist_params = params[n_mean+n_var:]
        
        return mean_params, variance_params, dist_params
    
    def _loglikelihood(self, params: np.ndarray) -> float:
        """Compute the negative log-likelihood for optimization.
        
        Parameters
        ----------
        params : np.ndarray
            Full parameter vector
            
        Returns
        -------
        float
            Negative log-likelihood
        """
        mean_params, variance_params, dist_params = self._split_params(params)
        
        # Compute residuals from mean model
        residuals = self.mean_model.compute_residuals(self.data, mean_params)
        
        # Compute conditional variances from variance model
        sigma2 = self.variance_model.compute_variance(residuals, variance_params)
        
        # Compute log-likelihood from distribution
        ll = self.distribution.loglikelihood(residuals, sigma2, dist_params)
        
        # Return negative log-likelihood for minimization
        return -ll
    
    def _get_starting_params(self) -> np.ndarray:
        """Get starting parameters for optimization.
        
        Returns
        -------
        np.ndarray
            Starting parameters
        """
        # Get starting parameters for mean model
        mean_params = self.mean_model.starting_params(self.data)
        
        # Compute residuals using mean model starting parameters
        residuals = self.mean_model.compute_residuals(self.data, mean_params)
        
        # Get starting parameters for variance model
        variance_params = self.variance_model.starting_params(residuals)
        
        # Compute conditional variances using variance model starting parameters
        sigma2 = self.variance_model.compute_variance(residuals, variance_params)
        
        # Compute standardized residuals
        std_residuals = residuals / np.sqrt(sigma2)
        
        # Get starting parameters for distribution
        dist_params = self.distribution.starting_params(std_residuals)
        
        # Combine all starting parameters
        return np.concatenate([mean_params, variance_params, dist_params])
    
    def _get_param_names(self) -> List[str]:
        """Get parameter names for all model components.
        
        Returns
        -------
        list
            List of parameter names
        """
        mean_names = self.mean_model.param_names()
        variance_names = self.variance_model.param_names()
        dist_names = self.distribution.param_names()
        
        return mean_names + variance_names + dist_names
    
    def fit(self, data: Union[np.ndarray, pd.Series], 
            method: str = 'BFGS', 
            options: Optional[Dict[str, Any]] = None) -> Results:
        """Fit the model to the data using maximum likelihood.
        
        Parameters
        ----------
        data : array_like
            Time series data to fit
        method : str, optional
            Optimization method for scipy.optimize.minimize
        options : dict, optional
            Options to pass to scipy.optimize.minimize
            
        Returns
        -------
        Results
            Estimation results
        """
        # Convert data to numpy array if needed
        if isinstance(data, pd.Series):
            data = data.values
        
        # Store data
        self.data = data
        self.nobs = len(data)
        
        # Get starting parameters
        start_params = self._get_starting_params()
        
        # Optimize the log-likelihood
        if options is None:
            options = {'disp': False}
        
        result = optimize.minimize(
            self._loglikelihood,
            start_params,
            method=method,
            options=options
        )
        
        # Extract optimized parameters
        params = result.x
        mean_params, variance_params, dist_params = self._split_params(params)
        
        # Compute residuals and conditional variances
        residuals = self.mean_model.compute_residuals(self.data, mean_params)
        sigma2 = self.variance_model.compute_variance(residuals, variance_params)
        std_residuals = residuals / np.sqrt(sigma2)
        
        # Compute log-likelihood
        loglikelihood = -result.fun
        
        # Get parameter names
        param_names = self._get_param_names()
        
        # Compute covariance matrix of parameters (Hessian inverse)
        try:
            hessian = optimize.approx_fprime(params, lambda p: optimize.approx_fprime(p, self._loglikelihood, 1e-4), 1e-4)
            cov_params = np.linalg.inv(hessian)
        except:
            cov_params = None
        
        # Create Results object
        results = Results(
            model=self,
            params=params,
            param_names=param_names,
            loglikelihood=loglikelihood,
            residuals=residuals,
            std_residuals=std_residuals,
            conditional_variance=sigma2,
            cov_params=cov_params
        )
        
        return results
    
    def simulate(self, nobs: int, params: Optional[np.ndarray] = None, 
                 burn: int = 100, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the model.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray, optional
            Parameters to use for simulation. If None, uses the estimated parameters.
        burn : int, optional
            Number of burn-in observations
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if params is None:
            raise ValueError("Parameters must be provided for simulation")
        
        mean_params, variance_params, dist_params = self._split_params(params)
        
        # Simulate with burn-in
        total_obs = nobs + burn
        
        # Simulate innovations
        innovations = self.distribution.simulate(total_obs, dist_params, rng=rng)
        
        # Simulate conditional variances
        sigma2 = self.variance_model.simulate(total_obs, variance_params, rng=rng)
        
        # Scale innovations by conditional standard deviations
        scaled_innovations = innovations * np.sqrt(sigma2)
        
        # Simulate mean process
        y = self.mean_model.simulate(total_obs, mean_params, errors=scaled_innovations, rng=rng)
        
        # Discard burn-in
        return y[burn:] 
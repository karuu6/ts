import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from typing import List, Optional, Union, Tuple, Any


class Distribution(ABC):
    """Base class for all distributions."""
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def loglikelihood(self, eps: np.ndarray, sigma2: np.ndarray, 
                      params: Optional[np.ndarray] = None) -> float:
        """Compute log-likelihood for the distribution.
        
        Parameters
        ----------
        eps : np.ndarray
            Standardized residuals
        sigma2 : np.ndarray
            Conditional variances
        params : np.ndarray, optional
            Distribution parameters
            
        Returns
        -------
        float
            Log-likelihood
        """
        pass
    
    @abstractmethod
    def simulate(self, nobs: int, params: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the distribution.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray, optional
            Distribution parameters
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        pass
    
    @abstractmethod
    def starting_params(self, standardized_residuals: np.ndarray) -> np.ndarray:
        """Compute starting parameters for the distribution.
        
        Parameters
        ----------
        standardized_residuals : np.ndarray
            Standardized residuals
            
        Returns
        -------
        np.ndarray
            Starting parameters
        """
        pass
    
    @abstractmethod
    def param_names(self) -> List[str]:
        """Get parameter names for the distribution.
        
        Returns
        -------
        list
            List of parameter names
        """
        pass
    
    @abstractmethod
    def num_params(self) -> int:
        """Get number of parameters in the distribution.
        
        Returns
        -------
        int
            Number of parameters
        """
        pass


class Normal(Distribution):
    """Normal distribution."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def loglikelihood(self, eps: np.ndarray, sigma2: np.ndarray, 
                      params: Optional[np.ndarray] = None) -> float:
        """Compute log-likelihood for the normal distribution.
        
        Parameters
        ----------
        eps : np.ndarray
            Standardized residuals
        sigma2 : np.ndarray
            Conditional variances
        params : np.ndarray, optional
            Distribution parameters (not used for normal)
            
        Returns
        -------
        float
            Log-likelihood
        """
        nobs = len(eps)
        ll = -0.5 * nobs * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2)) - 0.5 * np.sum(eps**2 / sigma2)
        return ll
    
    def simulate(self, nobs: int, params: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the normal distribution.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray, optional
            Distribution parameters (not used for normal)
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        if rng is None:
            rng = np.random.default_rng()
        
        return rng.standard_normal(nobs)
    
    def starting_params(self, standardized_residuals: np.ndarray) -> np.ndarray:
        """Compute starting parameters for the normal distribution.
        
        Parameters
        ----------
        standardized_residuals : np.ndarray
            Standardized residuals
            
        Returns
        -------
        np.ndarray
            Starting parameters (empty for normal)
        """
        return np.array([])
    
    def param_names(self) -> List[str]:
        """Get parameter names for the normal distribution.
        
        Returns
        -------
        list
            List of parameter names (empty for normal)
        """
        return []
    
    def num_params(self) -> int:
        """Get number of parameters in the normal distribution.
        
        Returns
        -------
        int
            Number of parameters (0 for normal)
        """
        return 0


class T(Distribution):
    """Student's t distribution."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def loglikelihood(self, eps: np.ndarray, sigma2: np.ndarray, params: np.ndarray) -> float:
        """Compute log-likelihood for the Student's t distribution.
        
        Parameters
        ----------
        eps : np.ndarray
            Standardized residuals
        sigma2 : np.ndarray
            Conditional variances
        params : np.ndarray
            Distribution parameters: [nu]
            
        Returns
        -------
        float
            Log-likelihood
        """
        nu = params[0]
        nobs = len(eps)
        
        # Compute log-likelihood for t distribution
        const = (
            stats.gamma.logpdf((nu + 1) / 2)
            - stats.gamma.logpdf(nu / 2)
            - 0.5 * np.log(np.pi * (nu - 2))
        )
        
        ll = nobs * const - 0.5 * np.sum(np.log(sigma2)) - (nu + 1) / 2 * np.sum(
            np.log(1 + eps**2 / (sigma2 * (nu - 2)))
        )
        
        return ll
    
    def simulate(self, nobs: int, params: np.ndarray, 
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the Student's t distribution.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray
            Distribution parameters: [nu]
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        nu = params[0]
        if rng is None:
            rng = np.random.default_rng()
        
        return stats.t.rvs(nu, size=nobs, random_state=rng)
    
    def starting_params(self, standardized_residuals: np.ndarray) -> np.ndarray:
        """Compute starting parameters for the Student's t distribution.
        
        Parameters
        ----------
        standardized_residuals : np.ndarray
            Standardized residuals
            
        Returns
        -------
        np.ndarray
            Starting parameters: [nu]
        """
        # Start with a reasonable degrees of freedom (e.g., 8)
        return np.array([8.0])
    
    def param_names(self) -> List[str]:
        """Get parameter names for the Student's t distribution.
        
        Returns
        -------
        list
            List of parameter names
        """
        return ['nu']
    
    def num_params(self) -> int:
        """Get number of parameters in the Student's t distribution.
        
        Returns
        -------
        int
            Number of parameters
        """
        return 1 
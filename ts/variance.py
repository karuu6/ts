import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any
from scipy.optimize import minimize
import warnings


class VarianceModel(ABC):
    """Base class for all variance models."""
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def simulate(self, nobs: int, params: np.ndarray, 
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the variance model.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray
            Model parameters
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated conditional variances
        """
        pass
    
    @abstractmethod
    def compute_variance(self, residuals: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute conditional variances for the variance model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from the mean model
        params : np.ndarray
            Model parameters
            
        Returns
        -------
        np.ndarray
            Conditional variances
        """
        pass
    
    @abstractmethod
    def starting_params(self, residuals: np.ndarray) -> np.ndarray:
        """Compute starting parameters for the variance model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from the mean model
            
        Returns
        -------
        np.ndarray
            Starting parameters
        """
        pass
    
    @abstractmethod
    def param_names(self) -> List[str]:
        """Get parameter names for the variance model.
        
        Returns
        -------
        list
            List of parameter names
        """
        pass
    
    @abstractmethod
    def num_params(self) -> int:
        """Get number of parameters in the variance model.
        
        Returns
        -------
        int
            Number of parameters
        """
        pass


class Constant(VarianceModel):
    """Constant variance model."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def simulate(self, nobs: int, params: np.ndarray, 
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return np.full(nobs, params[0])
    
    def compute_variance(self, residuals: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.full_like(residuals, params[0])
    
    def starting_params(self, residuals: np.ndarray) -> np.ndarray:
        return np.array([np.var(residuals)])
    
    def param_names(self) -> List[str]:
        return ['omega']
    
    def num_params(self) -> int:
        return 1


class GARCH(VarianceModel):
    """
    Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.
    
    The GARCH(p,q) model is defined as:
    σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    where:
    - ω (omega) is the constant term
    - α_i are the ARCH parameters for lagged squared errors
    - β_j are the GARCH parameters for lagged variances
    - p is the order of the GARCH terms
    - q is the order of the ARCH terms
    """
    
    def __init__(self, p=1, q=1):
        """
        Initialize GARCH model with specified orders.
        
        Parameters:
        -----------
        p : int
            Order of the GARCH terms (lagged variances)
        q : int
            Order of the ARCH terms (lagged squared errors)
        """
        super().__init__()
        self.p = p
        self.q = q
    
    def simulate(self, nobs: int, params: np.ndarray, errors: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate from a GARCH process.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray
            Model parameters: [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
        errors : np.ndarray, optional
            Innovations to use in simulation
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated conditional variances
        """
        if rng is None:
            rng = np.random.default_rng()
            
        # Extract parameters
        omega = params[0]
        alpha = params[1:1+self.q] if self.q > 0 else np.array([])
        beta = params[1+self.q:] if self.p > 0 else np.array([])
        
        # Generate innovations if not provided
        if errors is None:
            errors = rng.standard_normal(nobs)
            
        # Initialize arrays
        sigma2 = np.zeros(nobs)
        sigma2[0] = omega / (1 - np.sum(alpha) - np.sum(beta))  # Unconditional variance
        
        # Generate conditional variances
        for t in range(1, nobs):
            # ARCH component
            arch_term = 0
            for i in range(min(t, self.q)):
                if t-i-1 >= 0:
                    arch_term += alpha[i] * errors[t-i-1]**2
            
            # GARCH component
            garch_term = 0
            for j in range(min(t, self.p)):
                if t-j-1 >= 0:
                    garch_term += beta[j] * sigma2[t-j-1]
            
            # Combine components
            sigma2[t] = omega + arch_term + garch_term
        
        return sigma2
    
    def compute_variance(self, residuals: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute conditional variances for the GARCH model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from the mean model
        params : np.ndarray
            Model parameters: [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
            
        Returns
        -------
        np.ndarray
            Conditional variances
        """
        nobs = len(residuals)
        
        # Extract parameters
        omega = params[0]
        alpha = params[1:1+self.q] if self.q > 0 else np.array([])
        beta = params[1+self.q:] if self.p > 0 else np.array([])
        
        # Initialize arrays
        sigma2 = np.zeros(nobs)
        
        # Set initial variance to unconditional variance
        sigma2[0] = omega / (1 - np.sum(alpha) - np.sum(beta))
        
        # Compute conditional variances
        for t in range(1, nobs):
            # ARCH component
            arch_term = 0
            for i in range(min(t, self.q)):
                if t-i-1 >= 0:
                    arch_term += alpha[i] * residuals[t-i-1]**2
            
            # GARCH component
            garch_term = 0
            for j in range(min(t, self.p)):
                if t-j-1 >= 0:
                    garch_term += beta[j] * sigma2[t-j-1]
            
            # Combine components
            sigma2[t] = omega + arch_term + garch_term
        
        return sigma2
    
    def starting_params(self, residuals: np.ndarray) -> np.ndarray:
        """Compute starting parameters for the GARCH model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from the mean model
            
        Returns
        -------
        np.ndarray
            Starting parameters: [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
        """
        # Initialize parameters
        params = np.zeros(1 + self.p + self.q)
        
        # Set omega to a fraction of the unconditional variance
        params[0] = np.var(residuals) * 0.1  # omega
        
        # Set ARCH parameters
        if self.q > 0:
            params[1:1+self.q] = 0.05  # alpha
        
        # Set GARCH parameters
        if self.p > 0:
            params[1+self.q:] = 0.8  # beta
            
        return params
    
    def param_names(self) -> List[str]:
        """Get parameter names for the GARCH model.
        
        Returns
        -------
        list
            List of parameter names
        """
        names = ['omega']
        names.extend([f'alpha_{i+1}' for i in range(self.q)])
        names.extend([f'beta_{i+1}' for i in range(self.p)])
        return names
    
    def num_params(self) -> int:
        """Get number of parameters in the GARCH model.
        
        Returns
        -------
        int
            Number of parameters
        """
        return 1 + self.p + self.q 
import numpy as np
from abc import ABC, abstractmethod


class MeanModel(ABC):
    """Base class for all mean models."""
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def simulate(self, nobs, rng=None):
        """Simulate data from the mean model.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        pass
    
    @abstractmethod
    def compute_residuals(self, y, params):
        """Compute residuals for the mean model.
        
        Parameters
        ----------
        y : np.ndarray
            Observed data
        params : np.ndarray
            Model parameters
            
        Returns
        -------
        np.ndarray
            Residuals
        """
        pass
    
    @abstractmethod
    def starting_params(self, y):
        """Compute starting parameters for the mean model.
        
        Parameters
        ----------
        y : np.ndarray
            Observed data
            
        Returns
        -------
        np.ndarray
            Starting parameters
        """
        pass
    
    @abstractmethod
    def param_names(self):
        """Get parameter names for the mean model.
        
        Returns
        -------
        list
            List of parameter names
        """
        pass
    
    @abstractmethod
    def num_params(self):
        """Get number of parameters in the mean model.
        
        Returns
        -------
        int
            Number of parameters
        """
        pass


class Constant(MeanModel):
    """Constant mean model."""
    
    def __init__(self):
        super().__init__()
    
    def simulate(self, nobs, params, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        return np.full(nobs, params[0])
    
    def compute_residuals(self, y, params):
        return y - params[0]
    
    def starting_params(self, y):
        return np.array([np.mean(y)])
    
    def param_names(self):
        return ['mu']
    
    def num_params(self):
        return 1


class ARMA(MeanModel):
    """ARMA(p,q) mean model."""
    
    def __init__(self, p=1, q=0):
        super().__init__()
        self.p = p
        self.q = q
    
    def simulate(self, nobs, params, errors=None, rng=None):
        """Simulate from an ARMA process.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray
            Model parameters: [mu, phi_1, ..., phi_p, theta_1, ..., theta_q]
        errors : np.ndarray, optional
            Innovations to use in simulation
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        # Stub implementation - will be filled in later
        return np.zeros(nobs)
    
    def compute_residuals(self, y, params):
        """Compute residuals for the ARMA model.
        
        Parameters
        ----------
        y : np.ndarray
            Observed data
        params : np.ndarray
            Model parameters: [mu, phi_1, ..., phi_p, theta_1, ..., theta_q]
            
        Returns
        -------
        np.ndarray
            Residuals
        """
        # Stub implementation - will be filled in later
        return y - np.mean(y)
    
    def starting_params(self, y):
        """Compute starting parameters for the ARMA model.
        
        Parameters
        ----------
        y : np.ndarray
            Observed data
            
        Returns
        -------
        np.ndarray
            Starting parameters: [mu, phi_1, ..., phi_p, theta_1, ..., theta_q]
        """
        # Stub implementation - will be filled in later
        params = np.zeros(1 + self.p + self.q)
        params[0] = np.mean(y)
        return params
    
    def param_names(self):
        """Get parameter names for the ARMA model.
        
        Returns
        -------
        list
            List of parameter names
        """
        names = ['mu']
        names.extend([f'phi_{i+1}' for i in range(self.p)])
        names.extend([f'theta_{i+1}' for i in range(self.q)])
        return names
    
    def num_params(self):
        """Get number of parameters in the ARMA model.
        
        Returns
        -------
        int
            Number of parameters
        """
        return 1 + self.p + self.q 
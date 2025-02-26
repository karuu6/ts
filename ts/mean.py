import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any


class MeanModel(ABC):
    """Base class for all mean models."""
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def simulate(self, nobs: int, params: np.ndarray, errors: Optional[np.ndarray] = None, 
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate data from the mean model.
        
        Parameters
        ----------
        nobs : int
            Number of observations to simulate
        params : np.ndarray
            Model parameters
        errors : np.ndarray, optional
            Innovations to use in simulation
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Simulated data
        """
        pass
    
    @abstractmethod
    def compute_residuals(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
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
    def starting_params(self, y: np.ndarray) -> np.ndarray:
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
    def param_names(self) -> List[str]:
        """Get parameter names for the mean model.
        
        Returns
        -------
        list
            List of parameter names
        """
        pass
    
    @abstractmethod
    def num_params(self) -> int:
        """Get number of parameters in the mean model.
        
        Returns
        -------
        int
            Number of parameters
        """
        pass


class Constant(MeanModel):
    """Constant mean model."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def simulate(self, nobs: int, params: np.ndarray, errors: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        
        return np.full(nobs, params[0])
    
    def compute_residuals(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        return y - params[0]
    
    def starting_params(self, y: np.ndarray) -> np.ndarray:
        return np.array([np.mean(y)])
    
    def param_names(self) -> List[str]:
        return ['mu']
    
    def num_params(self) -> int:
        return 1


class ARMA(MeanModel):
    """ARMA(p,q) mean model."""
    
    def __init__(self, p: int = 1, q: int = 0) -> None:
        super().__init__()
        self.p = p
        self.q = q
    
    def simulate(self, nobs: int, params: np.ndarray, errors: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
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
        if rng is None:
            rng = np.random.default_rng()
        
        # Extract parameters
        mu = params[0]
        phi = params[1:1+self.p] if self.p > 0 else np.array([])
        theta = params[1+self.p:] if self.q > 0 else np.array([])
        
        # Generate innovations if not provided
        if errors is None:
            errors = rng.standard_normal(nobs)
        
        # Initialize arrays
        y = np.zeros(nobs)
        e = np.zeros(nobs)
        
        # Generate data
        max_lag = max(self.p, self.q)
        for t in range(nobs):
            # AR component
            ar_term = 0
            for i in range(min(t, self.p)):
                if t-i-1 >= 0:
                    ar_term += phi[i] * y[t-i-1]
            
            # MA component
            ma_term = 0
            for j in range(min(t, self.q)):
                if t-j-1 >= 0:
                    ma_term += theta[j] * e[t-j-1]
            
            # Combine components
            e[t] = errors[t]
            y[t] = mu + ar_term + ma_term + e[t]
        
        return y
    
    def compute_residuals(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
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
        nobs = len(y)
        
        # Extract parameters
        mu = params[0]
        phi = params[1:1+self.p] if self.p > 0 else np.array([])
        theta = params[1+self.p:] if self.q > 0 else np.array([])
        
        # Initialize arrays
        residuals = np.zeros(nobs)
        
        # Compute residuals
        for t in range(nobs):
            # Expected value based on AR component
            ar_term = 0
            for i in range(min(t, self.p)):
                if t-i-1 >= 0:
                    ar_term += phi[i] * y[t-i-1]
            
            # MA component using previous residuals
            ma_term = 0
            for j in range(min(t, self.q)):
                if t-j-1 >= 0:
                    ma_term += theta[j] * residuals[t-j-1]
            
            # Compute residual
            expected = mu + ar_term + ma_term
            residuals[t] = y[t] - expected
        
        return residuals
    
    def starting_params(self, y: np.ndarray) -> np.ndarray:
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
        # Initialize parameters
        params = np.zeros(1 + self.p + self.q)
        
        # Set mean
        params[0] = np.mean(y)
        
        # Set AR parameters using Yule-Walker equations if p > 0
        if self.p > 0:
            # Demean the data
            y_demean = y - params[0]
            
            # Compute autocorrelations
            acf = np.zeros(self.p + 1)
            for i in range(self.p + 1):
                if i == 0:
                    acf[i] = np.var(y_demean)
                else:
                    acf[i] = np.mean(y_demean[i:] * y_demean[:-i])
            
            # Solve Yule-Walker equations for AR parameters
            if self.p == 1:
                params[1] = acf[1] / acf[0]
            else:
                # For higher order AR models, use more sophisticated methods
                # This is a simple approximation
                for i in range(self.p):
                    params[i+1] = 0.5 / (i+1) if i < 2 else 0.1
        
        # Set MA parameters to small values
        if self.q > 0:
            params[1+self.p:] = 0.1
        
        return params
    
    def param_names(self) -> List[str]:
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
    
    def num_params(self) -> int:
        """Get number of parameters in the ARMA model.
        
        Returns
        -------
        int
            Number of parameters
        """
        return 1 + self.p + self.q 
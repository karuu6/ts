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
        self.omega = None
        self.alpha = None
        self.beta = None
        self.fitted = False
        
    def _log_likelihood(self, params, data):
        """
        Calculate the negative log-likelihood for GARCH model.
        
        Parameters:
        -----------
        params : array-like
            Model parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
        data : array-like
            Time series data (returns or residuals)
            
        Returns:
        --------
        float
            Negative log-likelihood value
        """
        T = len(data)
        omega = params[0]
        alpha = params[1:self.q+1]
        beta = params[self.q+1:]
        
        # Initialize arrays
        sigma2 = np.zeros(T)
        
        # Set initial variance to sample variance for stability
        sigma2[0] = np.var(data)
        
        # Recursively calculate conditional variances
        for t in range(1, T):
            # ARCH component (lagged squared errors)
            arch_component = 0
            for i in range(1, min(t+1, self.q+1)):
                if t-i >= 0:
                    arch_component += alpha[i-1] * data[t-i]**2
            
            # GARCH component (lagged variances)
            garch_component = 0
            for j in range(1, min(t+1, self.p+1)):
                if t-j >= 0:
                    garch_component += beta[j-1] * sigma2[t-j]
            
            # Calculate variance with a small constant to prevent zero/negative values
            sigma2[t] = max(1e-6, omega + arch_component + garch_component)
        
        # Calculate log-likelihood
        llh = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)
        
        return -llh  # Return negative log-likelihood for minimization
    
    def fit(self, data, method='SLSQP', max_iter=1000):
        """
        Fit the GARCH model to the data.
        
        Parameters:
        -----------
        data : array-like
            Time series data (returns or residuals)
        method : str
            Optimization method for scipy.optimize.minimize
        max_iter : int
            Maximum number of iterations for optimization
            
        Returns:
        --------
        self : object
            Fitted model instance
        """
        # Initial parameter guesses
        # Use reasonable defaults that ensure positive variance
        initial_omega = 0.01
        initial_alpha = np.ones(self.q) * 0.05
        initial_beta = np.ones(self.p) * 0.8
        
        initial_params = np.concatenate(([initial_omega], initial_alpha, initial_beta))
        
        # Parameter constraints
        # omega > 0, alpha_i >= 0, beta_j >= 0, sum(alpha_i + beta_j) < 1 for stationarity
        bounds = [(1e-6, None)] + [(0, 1)] * (self.p + self.q)
        
        # Constraint for stationarity: sum(alpha + beta) < 1
        constraint = {'type': 'ineq', 
                      'fun': lambda params: 1 - np.sum(params[1:])}
        
        # Optimize
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(data,),
            method=method,
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': max_iter}
        )
        
        if not result.success:
            warnings.warn(f"GARCH optimization did not converge: {result.message}")
        
        # Extract parameters
        self.omega = result.x[0]
        self.alpha = result.x[1:self.q+1]
        self.beta = result.x[self.q+1:]
        self.fitted = True
        
        return self
    
    def predict(self, data, n_steps=1):
        """
        Predict future conditional variances.
        
        Parameters:
        -----------
        data : array-like
            Historical time series data
        n_steps : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        array-like
            Predicted conditional variances
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        T = len(data)
        
        # Calculate historical variances
        hist_var = np.zeros(T)
        hist_var[0] = np.var(data)
        
        for t in range(1, T):
            # ARCH component
            arch_component = 0
            for i in range(1, min(t+1, self.q+1)):
                if t-i >= 0:
                    arch_component += self.alpha[i-1] * data[t-i]**2
            
            # GARCH component
            garch_component = 0
            for j in range(1, min(t+1, self.p+1)):
                if t-j >= 0:
                    garch_component += self.beta[j-1] * hist_var[t-j]
            
            hist_var[t] = max(1e-6, self.omega + arch_component + garch_component)
        
        # Forecast future variances
        forecasts = np.zeros(n_steps)
        
        # For a GARCH(1,1), the h-step ahead forecast is:
        # σ²_{T+h} = ω + (α + β)^{h-1} * (α * ε²_T + β * σ²_T)
        if self.p == 1 and self.q == 1:
            persistence = self.alpha[0] + self.beta[0]
            last_squared_error = data[-1]**2
            last_variance = hist_var[-1]
            
            for h in range(1, n_steps+1):
                if h == 1:
                    forecasts[h-1] = self.omega + self.alpha[0] * last_squared_error + self.beta[0] * last_variance
                else:
                    forecasts[h-1] = self.omega + persistence * forecasts[h-2]
        else:
            # For general GARCH(p,q), use recursive forecasting
            # This is a simplified approach for multi-step forecasting
            extended_var = np.append(hist_var, np.zeros(n_steps))
            extended_data = np.append(data, np.zeros(n_steps))
            
            for h in range(1, n_steps+1):
                t = T + h - 1
                
                # For forecasting, we replace future squared errors with their expected value (the variance)
                arch_component = 0
                for i in range(1, self.q+1):
                    if t-i >= T:  # Future value
                        arch_component += self.alpha[i-1] * extended_var[t-i]
                    else:  # Historical value
                        arch_component += self.alpha[i-1] * extended_data[t-i]**2
                
                garch_component = 0
                for j in range(1, self.p+1):
                    garch_component += self.beta[j-1] * extended_var[t-j]
                
                extended_var[t] = max(1e-6, self.omega + arch_component + garch_component)
                forecasts[h-1] = extended_var[t]
        
        return forecasts
    
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
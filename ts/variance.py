import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class VarianceModel(ABC):
    """Base class for all variance models."""

    @abstractmethod
    def __init__(self) -> None:
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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the variance model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Default implementation: no bounds
        return [(None, None)] * self.num_params()


class Constant(VarianceModel):
    """Constant variance model."""

    def __init__(self) -> None:
        super().__init__()

    def compute_variance(self, residuals: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.full_like(residuals, params[0])

    def starting_params(self, residuals: np.ndarray) -> np.ndarray:
        return np.array([np.var(residuals)])

    def param_names(self) -> List[str]:
        return ['omega']

    def num_params(self) -> int:
        return 1

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the constant variance model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Variance must be positive
        return [(1e-6, None)]


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
            params[1:1+self.q] = 0.4  # alpha

        # Set GARCH parameters
        if self.p > 0:
            params[1+self.q:] = 0.4  # beta

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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the GARCH model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # omega must be positive
        bounds = [(1e-6, None)]

        # alpha and beta parameters must be positive for a valid GARCH model
        # and their sum should be less than 1 for stationarity
        # We'll enforce individual constraints here
        bounds.extend([(0.0, 1.0) for _ in range(self.q)])  # alpha
        bounds.extend([(0.0, 1.0) for _ in range(self.p)])  # beta

        return bounds

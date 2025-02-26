import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class MeanModel(ABC):
    """Base class for all mean models."""

    @abstractmethod
    def __init__(self) -> None:
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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the mean model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Default implementation: no bounds
        return [(None, None)] * self.num_params()


class Constant(MeanModel):
    """Constant mean model."""

    def __init__(self) -> None:
        super().__init__()

    def compute_residuals(self, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        return y - params[0]

    def starting_params(self, y: np.ndarray) -> np.ndarray:
        return np.array([np.mean(y)])

    def param_names(self) -> List[str]:
        return ['mu']

    def num_params(self) -> int:
        return 1

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the constant mean model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # No specific bounds for the constant term
        return [(None, None)]


class ARMA(MeanModel):
    """ARMA(p,q) mean model."""

    def __init__(self, p: int = 1, q: int = 0) -> None:
        super().__init__()
        self.p = p
        self.q = q

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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the ARMA model.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # No specific bounds for mu
        bounds = [(None, None)]

        # For AR parameters, typically we want to ensure stationarity
        # A simple approximation is to constrain each phi between -1 and 1
        bounds.extend([(-0.99, 0.99) for _ in range(self.p)])

        # For MA parameters, typically we want to ensure invertibility
        # A simple approximation is to constrain each theta between -1 and 1
        bounds.extend([(-0.99, 0.99) for _ in range(self.q)])

        return bounds

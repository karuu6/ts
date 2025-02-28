import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from typing import List, Optional, Tuple


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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the distribution.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Default implementation: no bounds
        return [(None, None)] * self.num_params()


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
        ll = -0.5 * nobs * \
            np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2)) - \
            0.5 * np.sum(eps**2 / sigma2)
        return ll

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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the normal distribution.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Normal distribution has no parameters
        return []


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

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for parameters in the Student's t distribution.

        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each parameter
        """
        # Degrees of freedom must be > 2 for finite variance
        return [(2.1, None)]

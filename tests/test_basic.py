import numpy as np
import pandas as pd
import pytest
from scipy import stats
from typing import List, Optional, Tuple, Dict, Any, Union

from ts import (
    UnivariateTimeSeries,
    ConstantMean,
    ARMA,
    ConstantVariance,
    GARCH,
    Normal,
    T
)


def test_import() -> None:
    """Test that all modules can be imported."""
    # This test passes if the imports above don't fail
    pass


def test_normal_distribution() -> None:
    """Test the normal distribution."""
    # Create a normal distribution
    dist = Normal()

    # Test parameter names and counts
    assert dist.param_names() == []
    assert dist.num_params() == 0

    # Test starting parameters
    std_residuals = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    params = dist.starting_params(std_residuals)
    assert len(params) == 0

    # Test log-likelihood computation
    eps = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    sigma2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ll = dist.loglikelihood(eps, sigma2)

    # Expected log-likelihood for standard normal
    expected_ll = -0.5 * len(eps) * np.log(2 * np.pi) - 0.5 * np.sum(eps**2)
    assert np.isclose(ll, expected_ll)


def test_fit_constant_model() -> None:
    """Test fitting a constant mean and variance model."""
    # Generate some data
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=2.0, size=100)
    # Create and fit the model
    model = UnivariateTimeSeries()
    results = model.fit(data)

    # Check results
    assert isinstance(results.params, np.ndarray)
    assert len(results.params) == 2  # mu and omega
    assert np.isclose(results.params[0], np.mean(
        data), rtol=0.1)  # mu should be close to mean
    # omega should be close to variance
    assert np.isclose(results.params[1], np.var(data), rtol=0.1)

    # Check other results attributes
    # Log-likelihood should be negative for this model
    assert results.loglikelihood < 0
    assert len(results.residuals) == len(data)
    assert len(results.std_residuals) == len(data)
    assert results.aic is not None
    assert results.bic is not None


def test_fit_arma_model() -> None:
    """Test fitting an ARMA model with constant variance."""
    # Generate some AR(1) data
    np.random.seed(42)
    n = 300
    phi = 0.9
    data = np.zeros(n)
    data[0] = np.random.normal()
    for t in range(1, n):
        data[t] = phi * data[t-1] + np.random.normal()

    # Create and fit the model
    model = UnivariateTimeSeries(mean_model=ARMA(
        p=1, q=0), variance_model=ConstantVariance())
    results = model.fit(data)

    # Check results
    assert isinstance(results.params, np.ndarray)
    assert len(results.params) == 3  # mu, phi, omega
    # phi should be close to 0.9
    assert np.isclose(results.params[1], phi, rtol=0.2)


def test_fit_garch_model() -> None:
    """Test fitting a GARCH model with constant mean."""
    # Generate some GARCH(1,1) data
    # np.random.seed(42)
    n = 3000
    omega = 0.1
    alpha = 0.6
    beta = 0.2

    data = np.zeros(n)
    variances = np.zeros(n)

    variances[0] = omega / (1 - alpha - beta)
    z = np.random.normal(0, 1, n)

    # Simulate the GARCH(1,1) process
    for t in range(1, n):
        variances[t] = omega + alpha * data[t-1]**2 + beta * variances[t-1]
        data[t] = np.sqrt(variances[t]) * z[t]

    # Create and fit the model
    model = UnivariateTimeSeries(
        mean_model=ConstantMean(), variance_model=GARCH(p=1, q=1))
    results = model.fit(data)
    results.summary()

    # Check results
    assert isinstance(results.params, np.ndarray)
    assert len(results.params) == 4  # mu, omega, alpha, beta
    assert np.isclose(results.params[1], omega, rtol=0.2)
    assert np.isclose(results.params[2], alpha, rtol=0.2)
    assert np.isclose(results.params[3], beta, rtol=0.2)

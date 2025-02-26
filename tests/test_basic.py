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


def test_constant_mean() -> None:
    """Test the constant mean model."""
    # Create a constant mean model
    mean_model = ConstantMean()
    
    # Test parameter names and counts
    assert mean_model.param_names() == ['mu']
    assert mean_model.num_params() == 1
    
    # Test starting parameters
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    params = mean_model.starting_params(data)
    assert len(params) == 1
    assert params[0] == 3.0  # mean of data
    
    # Test residuals computation
    residuals = mean_model.compute_residuals(data, np.array([3.0]))
    expected_residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(residuals, expected_residuals)
    
    # Test simulation
    simulated = mean_model.simulate(10, np.array([3.0]))
    assert len(simulated) == 10
    assert np.all(simulated == 3.0)


def test_constant_variance() -> None:
    """Test the constant variance model."""
    # Create a constant variance model
    var_model = ConstantVariance()
    
    # Test parameter names and counts
    assert var_model.param_names() == ['omega']
    assert var_model.num_params() == 1
    
    # Test starting parameters
    residuals = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    params = var_model.starting_params(residuals)
    assert len(params) == 1
    assert params[0] == np.var(residuals)  # variance of residuals
    
    # Test variance computation
    variance = var_model.compute_variance(residuals, np.array([4.0]))
    expected_variance = np.full_like(residuals, 4.0)
    np.testing.assert_array_almost_equal(variance, expected_variance)
    
    # Test simulation
    simulated = var_model.simulate(10, np.array([4.0]))
    assert len(simulated) == 10
    assert np.all(simulated == 4.0)


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
    
    # Test simulation
    simulated = dist.simulate(100)
    assert len(simulated) == 100
    # Check that simulated data has approximately normal distribution
    _, p_value = stats.normaltest(simulated)
    assert p_value > 0.01  # Not a strict test, but should catch major issues


def test_univariate_time_series_basic() -> None:
    """Test basic functionality of UnivariateTimeSeries class."""
    # Create a simple model with constant mean and variance
    model = UnivariateTimeSeries()
    
    # Check default models
    assert isinstance(model.mean_model, ConstantMean)
    assert isinstance(model.variance_model, ConstantVariance)
    assert isinstance(model.distribution, Normal)
    
    # Test parameter splitting
    params = np.array([1.0, 2.0])  # [mean, variance]
    mean_params, var_params, dist_params = model._split_params(params)
    assert len(mean_params) == 1
    assert mean_params[0] == 1.0
    assert len(var_params) == 1
    assert var_params[0] == 2.0
    assert len(dist_params) == 0
    
    # Test parameter names
    param_names = model._get_param_names()
    assert param_names == ['mu', 'omega']


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
    assert np.isclose(results.params[0], np.mean(data), rtol=0.1)  # mu should be close to mean
    assert np.isclose(results.params[1], np.var(data), rtol=0.1)   # omega should be close to variance
    
    # Check other results attributes
    assert results.loglikelihood < 0  # Log-likelihood should be negative for this model
    assert len(results.residuals) == len(data)
    assert len(results.std_residuals) == len(data)
    assert len(results.conditional_variance) == len(data)
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
    model = UnivariateTimeSeries(mean_model=ARMA(p=1, q=0), variance_model=ConstantVariance())
    results = model.fit(data)
    
    # Check results
    assert isinstance(results.params, np.ndarray)
    assert len(results.params) == 3  # mu, phi, omega
    assert np.isclose(results.params[1], phi, rtol=0.2)  # phi should be close to 0.9 

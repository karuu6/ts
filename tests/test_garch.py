import numpy as np
import pytest
from ts.variance import GARCH

def test_garch_initialization():
    """Test GARCH model initialization with different orders."""
    # Test default initialization
    garch11 = GARCH()
    assert garch11.p == 1
    assert garch11.q == 1
    
    # Test custom orders
    garch22 = GARCH(p=2, q=2)
    assert garch22.p == 2
    assert garch22.q == 2

def test_garch_fit_predict():
    """Test GARCH model fitting and prediction on simulated data."""
    # Set random seed for reproducibility
    # np.random.seed(42)
    
    # Simulate GARCH(1,1) process
    T = 1000
    omega, alpha, beta = 0.01, 0.1, 0.8
    
    # Initialize arrays
    epsilon = np.zeros(T)
    sigma2 = np.zeros(T)
    
    # Set initial variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate GARCH process
    for t in range(1, T):
        epsilon[t-1] = np.sqrt(sigma2[t-1]) * np.random.normal(0, 1)
        sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
    
    # Generate the last observation
    epsilon[-1] = np.sqrt(sigma2[-1]) * np.random.normal(0, 1)
    
    # Fit GARCH model
    model = GARCH(p=1, q=1)
    model.fit(epsilon)
    
    # Check if parameters are reasonable
    assert model.omega > 0
    assert 0 <= model.alpha[0] <= 1
    assert 0 <= model.beta[0] <= 1
    assert model.alpha[0] + model.beta[0] < 1  # Stationarity condition
    
    # Check if parameters are close to the true values (with some tolerance)
    assert abs(model.omega - omega) < 0.05
    assert abs(model.alpha[0] - alpha) < 0.1
    assert abs(model.beta[0] - beta) < 0.1
    
    # Test prediction
    forecast = model.predict(epsilon, n_steps=5)
    assert len(forecast) == 5
    assert np.all(forecast > 0)  # Variances should be positive

def test_garch_higher_order():
    """Test higher-order GARCH model."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random returns
    returns = np.random.normal(0, 1, 500)
    
    # Fit GARCH(2,2) model
    model = GARCH(p=2, q=2)
    model.fit(returns)
    
    # Check parameters
    assert model.omega > 0
    assert len(model.alpha) == 2
    assert len(model.beta) == 2
    assert np.all(model.alpha >= 0)
    assert np.all(model.beta >= 0)
    assert np.sum(model.alpha) + np.sum(model.beta) < 1  # Stationarity condition
    
    # Test prediction
    forecast = model.predict(returns, n_steps=10)
    assert len(forecast) == 10
    assert np.all(forecast > 0)

def test_garch_numerical_stability():
    """Test GARCH model with extreme values to check numerical stability."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data with extreme values
    T = 200
    data = np.random.normal(0, 1, T)
    
    # Add some extreme values
    data[50] = 10.0  # Large positive shock
    data[100] = -10.0  # Large negative shock
    data[150] = 0.0  # Zero value
    
    # Fit GARCH model
    model = GARCH()
    model.fit(data)
    
    # Check if model fitting succeeded
    assert model.fitted
    
    # Test prediction
    forecast = model.predict(data, n_steps=5)
    assert len(forecast) == 5
    assert np.all(forecast > 0)
    assert np.all(np.isfinite(forecast))  # Check for NaN or Inf values 
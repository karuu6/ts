from arch import arch_model
import numpy as np
from ts import (
    UnivariateTimeSeries,
    ConstantMean,
    ARMA,
    ConstantVariance,
    GARCH,
    Normal,
    T
)


def test_arch_library() -> None:
    """Test against the arch library."""
    # Generate some GARCH(1,1) data
    # np.random.seed(42)
    n = 3000
    omega = 0.1
    alpha = 0.3
    beta = 0.5

    data = np.zeros(n)
    variances = np.zeros(n)

    variances[0] = omega / (1 - alpha - beta)
    z = np.random.normal(0, 1, n)

    # Simulate the GARCH(1,1) process
    for t in range(1, n):
        variances[t] = omega + alpha * data[t-1]**2 + beta * variances[t-1]
        data[t] = np.sqrt(variances[t]) * z[t]

    # Create and fit the model
    model = arch_model(data, mean='Constant', vol='GARCH', p=1, q=1)
    results = model.fit()

    model2 = UnivariateTimeSeries(
        mean_model=ConstantMean(), variance_model=GARCH(p=1, q=1))
    results2 = model2.fit(data)

    # Check results
    assert np.isclose(results.params[1], results2.params[1], rtol=0.1)
    assert np.isclose(results.params[2], results2.params[2], rtol=0.1)
    assert np.isclose(results.params[3], results2.params[3], rtol=0.1)

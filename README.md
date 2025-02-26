# Time Series Analysis Library

A Python library for time series analysis, modeling, and forecasting.

## Features

- Univariate time series modeling
- Supports GARCH models for conditional variance
- Supports ARMA models for conditional mean
- Supports Joint estimation of mean and variance models using MLE
- Various distribution options for error terms

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/karuu6/ts.git
```

## Usage

```python
from ts import UnivariateTimeSeries

# Create a time series model
model = UnivariateTimeSeries(mean_model=ARMA(p=1, q=1), variance_model=GARCH(p=1, q=1))

# Fit the model to some data
data = ...
results = model.fit(data)

# Print a summary of the results
results.summary()
```

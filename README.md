# DeepTCN

deeptcn is a python package with an unofficial implementation of the DeepTCN probabilistic forecasting model presented in the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220303441) "Probabilistic forecasting with temporal convolutional neural network" by Chen, Yitian, et al.

deeptcn supports both gaussian and quantile loss functions, past and future covariates, univariates and multivariates time series.

This implementation is based on the Pytorch Lightning framework.

# Installation

Install deeptcn using pip as follows:

`pip install git+https://github.com/vpozdnyakov/deeptcn.git`

# Quick start

```python
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
from deeptcn import GaussianDeepTCN, utils

# Download the data
electricity_train, electricity_test = utils.electricity_dataset()

# Select columns to forecast
input_len, output_len = 104, 64
train_target = electricity_train.iloc[:, :3]
test_target = electricity_test.iloc[:, :3]
past_target = test_target.iloc[:input_len]
future_target = test_target.iloc[input_len:]

# Train the model
model = GaussianDeepTCN(input_len, output_len)
model.fit(train_target.values)

# Forecast with the 80% confidence interval
pred_mu, pred_sigma = model.predict(past_target.values)
pred_q01 = norm.ppf(q=0.1, loc=pred_mu, scale=pred_sigma)
pred_q09 = norm.ppf(q=0.9, loc=pred_mu, scale=pred_sigma)
print(f'Pinball loss, alpha 0.1: {mean_pinball_loss(future_target, pred_q01, alpha=0.1):.4f}')
print(f'Pinball loss, alpha 0.9: {mean_pinball_loss(future_target, pred_q09, alpha=0.9):.4f}')

# Pinball loss, alpha 0.1: 0.0160
# Pinball loss, alpha 0.9: 0.0143
```

<img src='https://raw.githubusercontent.com/vpozdnyakov/deeptcn/main/images/forecast_1.png' width=600>

<img src='https://raw.githubusercontent.com/vpozdnyakov/deeptcn/main/images/forecast_2.png' width=600>

<img src='https://raw.githubusercontent.com/vpozdnyakov/deeptcn/main/images/forecast_3.png' width=600>

More examples here [electricity.ipynb](examples/electricity.ipynb)


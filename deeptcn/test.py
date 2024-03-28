import pandas as pd
from deeptcn import GaussianDeepTCN, QuantileDeepTCN


class TestOnElectricity:
    def setup_class(self):
        self.input_len = 104
        self.output_len = 64
        electricity_train = pd.read_csv('datasets/electricity/electricity_train.csv', index_col=0)
        electricity_train.index = pd.to_datetime(electricity_train.index)
        electricity_test = pd.read_csv('datasets/electricity/electricity_test.csv', index_col=0)
        electricity_test.index = pd.to_datetime(electricity_test.index)
        self.train_target, self.train_cov = electricity_train.iloc[:, :370], electricity_train.iloc[:, 372:374]
        self.test_target, self.test_cov = electricity_test.iloc[:, :370], electricity_test.iloc[:, 372:374]
        self.past_target = self.test_target.iloc[:self.input_len]
        self.future_cov = self.test_cov.iloc[self.input_len:]
        self.future_target = self.test_target.iloc[self.input_len:]
        
    def test_gaussian(self):
        model = GaussianDeepTCN(
            input_len=self.input_len,
            output_len=self.output_len,
            hidden_dim=64,
            dropout=0.1,
            kernel_size=3, 
            num_layers=2, 
            lr=0.001, 
            batch_size=32, 
            num_epochs=1, 
            verbose=True,
            validation_size=0.2,
            accelerator='cpu',
        )
        model.fit(self.train_target.values, future_covariates=self.train_cov.values)
        pred_mu, _ = model.predict(self.past_target.values, future_covariates=self.future_cov.values)
        assert pred_mu.shape == self.future_target.shape

    def test_quantile(self):
        model = QuantileDeepTCN(
            input_len=self.input_len,
            output_len=self.output_len,
            hidden_dim=64,
            dropout=0.1,
            kernel_size=3, 
            num_layers=2, 
            lr=0.001, 
            batch_size=32, 
            num_epochs=1, 
            verbose=True,
            validation_size=0.2,
            quantiles=[0.1, 0.5, 0.8],
            accelerator='cpu',
        )
        model.fit(self.train_target.values, future_covariates=self.train_cov.values)
        pred_q01, _, _ = model.predict(self.past_target.values, future_covariates=self.future_cov.values)
        assert pred_q01.shape == self.future_target.shape

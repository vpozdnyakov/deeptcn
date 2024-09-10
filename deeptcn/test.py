import pandas as pd
from deeptcn import GaussianDeepTCN, QuantileDeepTCN
import pytest

testparams = [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
]

class TestOnElectricity:
    def setup_class(self):
        self.input_len = 104
        self.output_len = 64
        electricity_train = pd.read_csv('datasets/electricity/electricity_train.csv', index_col=0)
        electricity_train.index = pd.to_datetime(electricity_train.index)
        electricity_test = pd.read_csv('datasets/electricity/electricity_test.csv', index_col=0)
        electricity_test.index = pd.to_datetime(electricity_test.index)
        self.train_target, self.train_past_cov, self.train_future_cov = (
            electricity_train.iloc[:, 350:370], 
            electricity_train.iloc[:, :350], 
            electricity_train.iloc[:, 372:374]
        )
        self.test_target, self.test_past_cov, self.test_future_cov = (
            electricity_test.iloc[:, 350:370],
            electricity_test.iloc[:, :350],  
            electricity_test.iloc[:, 372:374]
        )
        self.past_target = self.test_target.iloc[:self.input_len]
        self.past_cov = self.test_past_cov.iloc[:self.input_len]
        self.future_cov = self.test_future_cov.iloc[self.input_len:]
        self.future_target = self.test_target.iloc[self.input_len:]
    
    @pytest.mark.parametrize("with_past_cov,with_future_cov", testparams)
    def test_gaussian(self, with_past_cov, with_future_cov):
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
        train_past_cov = self.train_past_cov.values if with_past_cov else None
        train_future_cov = self.train_future_cov.values if with_future_cov else None
        model.fit(self.train_target.values, past_covariates=train_past_cov, future_covariates=train_future_cov)
        past_cov = self.past_cov.values if with_past_cov else None
        future_cov = self.future_cov.values if with_future_cov else None
        pred_mu, _ = model.predict(self.past_target.values, past_covariates=past_cov, future_covariates=future_cov)
        assert pred_mu.shape == self.future_target.shape

    @pytest.mark.parametrize("with_past_cov,with_future_cov", testparams)
    def test_quantile(self, with_past_cov, with_future_cov):
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
        train_past_cov = self.train_past_cov.values if with_past_cov else None
        train_future_cov = self.train_future_cov.values if with_future_cov else None
        model.fit(self.train_target.values, past_covariates=train_past_cov, future_covariates=train_future_cov)
        past_cov = self.past_cov.values if with_past_cov else None
        future_cov = self.future_cov.values if with_future_cov else None
        pred_q01, _, _ = model.predict(self.past_target.values, past_covariates=past_cov, future_covariates=future_cov)
        assert pred_q01.shape == self.future_target.shape

    def test_loading(self):
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
        model.fit(self.train_target.values)
        default_root_dir = model.trainer.default_root_dir
        version = model.trainer.logger.version
        checkpoint_path = f'{default_root_dir}/lightning_logs/version_{version}/checkpoints/epoch=0-step=23.ckpt'
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
        model.load_from_checkpoint(checkpoint_path, self.train_target.values)
        pred_q01, _, _ = model.predict(self.past_target.values)
        assert pred_q01.shape == self.future_target.shape

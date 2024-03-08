from deeptcn.base import DeepTCN, DeepTCNModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDeepTCNModule(DeepTCNModule):
    def _create_output(self):
        self.output = nn.Linear(self.hidden_dim, self.target_dim*2)
    
    def forward(self, past_target, past_cov=None, future_cov=None):
        # batch_size, seq_len, input_dim
        h = self.hidden_state(past_target, past_cov, future_cov)
        output = self.output(h)
        mu = output[..., :self.target_dim]
        sigma = F.relu(output[..., self.target_dim:])
        return mu, sigma

    def training_step(self, batch, batch_idx):
        past_cov, past_target, future_cov, future_target = batch
        mu, sigma = self.forward(past_target, past_cov, future_cov)
        loss = F.gaussian_nll_loss(input=mu, target=future_target, var=sigma**2, full=True)
        self.log("train_loss", loss)
        return loss


class GaussianDeepTCN(DeepTCN):
    """
    DeepTCN probabilistic model, the implementation is based on the paper Chen, 
    Yitian, et al. "Probabilistic forecasting with temporal convolutional neural 
    network." Neurocomputing 399 (2020): 491-501.
    """
    def __init__(
            self, 
            input_len, 
            output_len, 
            hidden_dim: int=128,
            dropout: float=0.,
            kernel_size: int=3,
            num_layers: int=2,
            lr: float=0.001,
            batch_size=32,
            num_epochs=10,
            verbose=False,
        ):
        super().__init__(
            input_len, output_len, hidden_dim, dropout, kernel_size, num_layers, 
            lr, batch_size, num_epochs, verbose
        )

    def _create_model(self, target, past_covariates=None, future_covariates=None):
        self.model = GaussianDeepTCNModule(
            target_dim=len(target.columns),
            past_cov_dim=len(past_covariates.columns) if past_covariates else 0,
            future_cov_dim=len(future_covariates.columns) if future_covariates else 0,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            lr=self.lr,
            uses_past_covariates=past_covariates is not None,
            uses_future_covariates=future_covariates is not None,
        )

    def predict(self, past_target, past_covariates=None, future_covariates=None):
        past_target, past_covariates, future_covariates = self._preprocess_input(
            past_target, past_covariates, future_covariates)
        with torch.no_grad():
            mu, _ = self.model(past_target, past_covariates, future_covariates)
        mu = mu[0, -self.output_len:, :]
        return mu.cpu().numpy()
    
    def sample(self, past_target, past_covariates=None, future_covariates=None, n_samples=100):
        past_target, past_covariates, future_covariates = self._preprocess_input(
            past_target, past_covariates, future_covariates)
        with torch.no_grad():
            mu, sigma = self.model(past_target, past_covariates, future_covariates)
        noise = torch.randn(n_samples, mu.shape[1], mu.shape[2])
        sample = mu + noise * sigma
        sample = sample[:, -self.output_len:, :]
        return sample.cpu().numpy()

from deeptcn.base import DeepTCN, DeepTCNModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileDeepTCNModule(DeepTCNModule):
    def __init__(
            self, 
            target_dim: int, 
            past_cov_dim: int, 
            future_cov_dim: int, 
            hidden_dim: int, 
            kernel_size: int, 
            dropout: float, 
            num_layers: int, 
            quantiles: list[int],
            lr: float, 
            uses_past_covariates: bool, 
            uses_future_covariates: bool
        ):
        self.quantiles = quantiles
        super().__init__(target_dim, past_cov_dim, future_cov_dim, hidden_dim, kernel_size, dropout, num_layers, lr, uses_past_covariates, uses_future_covariates)

    def _create_output(self):
        self.output = nn.Linear(self.hidden_dim, self.target_dim * len(self.quantiles))
    
    def forward(self, past_target, past_cov=None, future_cov=None):
        # batch_size, seq_len, input_dim
        h = self.hidden_state(past_target, past_cov, future_cov)
        output = self.output(h)
        qvalues = output.reshape(*output.size()[:2], self.target_dim, len(self.quantiles))
        return qvalues
    
    def quantile_loss(self, qvalues, target, quantiles):
        upper = F.relu(qvalues - target[..., None]) * (1 - quantiles)
        lower = F.relu(target[..., None] - qvalues) * quantiles
        loss = (upper + lower).sum(dim=(1,2,3))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        past_cov, past_target, future_cov, future_target = batch
        qvalues = self.forward(past_target, past_cov, future_cov)
        quantiles = torch.tensor(self.quantiles, device=self.device)
        quantiles = quantiles[None, None, None, :] # reshaping
        loss = self.quantile_loss(qvalues, future_target, quantiles)
        self.log("train_loss", loss)
        return loss


class QuantileDeepTCN(DeepTCN):
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
            quantiles: list[int]=[0.5, 0.9],
            lr: float=0.001,
            batch_size=32,
            num_epochs=10,
            verbose=False,
        ):
        super().__init__(
            input_len, output_len, hidden_dim, dropout, kernel_size, num_layers, 
            lr, batch_size, num_epochs, verbose
        )
        self.quantiles = quantiles

    def _create_model(self, target, past_covariates=None, future_covariates=None):
        self.model = QuantileDeepTCNModule(
            target_dim=len(target.columns),
            past_cov_dim=len(past_covariates.columns) if past_covariates else 0,
            future_cov_dim=len(future_covariates.columns) if future_covariates else 0,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            quantiles=self.quantiles,
            lr=self.lr,
            uses_past_covariates=past_covariates is not None,
            uses_future_covariates=future_covariates is not None,
        )

    def predict(self, past_target, past_covariates=None, future_covariates=None):
        past_target, past_covariates, future_covariates = self._preprocess_input(
            past_target, past_covariates, future_covariates)
        with torch.no_grad():
            q = self.model(past_target, past_covariates, future_covariates)
        q = q[0, -self.output_len:, :]
        return q.cpu().numpy()

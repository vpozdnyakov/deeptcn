from deeptcn.base import DeepTCN, DeepTCNModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDeepTCNModule(DeepTCNModule):
    def create_output(self):
        self.output = nn.Linear(self.hidden_dim, self.target_dim*2)
    
    def forward(self, past_target, past_cov=None, future_cov=None):
        # batch_size, seq_len, input_dim
        h = self.hidden_state(past_target, past_cov, future_cov)
        output = self.output(h)
        output = output.reshape(*output.size()[:2], self.target_dim, 2)
        mu = output[..., 0]
        sigma = F.softplus(output[..., 1])
        return mu, sigma
    
    def calc_loss(self, batch):
        past_cov, past_target, future_cov, future_target = batch
        mu, sigma = self.forward(past_target, past_cov, future_cov)
        return F.gaussian_nll_loss(input=mu, target=future_target, var=sigma**2, full=True)

class GaussianDeepTCN(DeepTCN):
    """
    DeepTCN probabilistic model.
    """
    def __init__(
            self, 
            input_len: int, 
            output_len: int, 
            hidden_dim: int=64,
            dropout: float=0.,
            kernel_size: int=3,
            num_layers: int=2,
            lr: float=0.001,
            batch_size: int=32,
            num_epochs: int=10,
            verbose: bool=False,
            accelerator: str='auto',
            validation_size: float=0.,
        ):
        super().__init__(
            input_len, output_len, hidden_dim, dropout, kernel_size, num_layers, 
            lr, batch_size, num_epochs, verbose, accelerator, validation_size
        )

    def _create_model(self):
        self.model = GaussianDeepTCNModule(
            target_dim=self.target_dim,
            past_cov_dim=self.past_cov_dim,
            future_cov_dim=self.future_cov_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            lr=self.lr,
            with_past_covariates=self.with_past_covariates,
            with_future_covariates=self.with_future_covariates,
        )

    def predict(self, past_target, past_covariates=None, future_covariates=None):
        """
        Predict/forecast the given time series.

        Args:
            past_target: Past target time series. Array of the shape (n, m) 
                where n is the input length and m is the number of components.
            past_covariates: Past covariates. Array of the shape (n, m) where n 
                is the input length and m is the number of components.
            future_covariates: Future covariates. Array of the shape (n, m) 
                where n is the output length and m is the number of components.
        
        Returns:
            List[np.ndarray]: Mu and sigma represent distribution of the forecast. 
                The both arrays are of the shape (n, m) where n is the output 
                length and m is the number of components.
        """
        past_target, past_covariates, future_covariates = self._preprocess_input(
            past_target, past_covariates, future_covariates)
        with torch.no_grad():
            mu, sigma = self.model(past_target, past_covariates, future_covariates)
        mu = mu[0, -self.output_len:, :]
        sigma = sigma[0, -self.output_len:, :]
        return [mu.cpu().numpy(), sigma.cpu().numpy()]

from deeptcn.base import DeepTCN, DeepTCNModule
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_loss(qvalues, target, quantiles):
    """
    Quantile loss corresponds to Eq. 6 in the main paper.
    """
    upper = F.relu(qvalues - target) * (1 - quantiles)
    lower = F.relu(target - qvalues) * quantiles
    loss = 2 * (upper + lower).sum(dim=(1,2,3))
    return loss.mean()


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
            with_past_covariates: bool,
            with_future_covariates: bool,
        ):
        self.quantiles = quantiles
        super().__init__(
            target_dim, past_cov_dim, future_cov_dim, hidden_dim, kernel_size, 
            dropout, num_layers, lr, with_past_covariates, with_future_covariates,
        )

    def create_output(self):
        self.output = nn.Linear(self.hidden_dim, self.target_dim * len(self.quantiles))
    
    def forward(self, past_target, past_cov=None, future_cov=None):
        # batch_size, seq_len, input_dim
        h = self.hidden_state(past_target, past_cov, future_cov)
        output = self.output(h)
        qvalues = output.reshape(*output.size()[:2], self.target_dim, len(self.quantiles))
        return qvalues

    def calc_loss(self, batch):
        past_cov, past_target, future_cov, future_target = batch
        qvalues = self.forward(past_target, past_cov, future_cov)
        quantiles = torch.tensor(self.quantiles, device=self.device)
        future_target = future_target[..., None] #reshaping
        quantiles = quantiles[None, None, None, :] # reshaping
        return quantile_loss(qvalues, future_target, quantiles)


class QuantileDeepTCN(DeepTCN):
    """
    Quantile DeepTCN probabilistic model.
    """
    def __init__(
            self, 
            input_len: int, 
            output_len: int, 
            hidden_dim: int=64,
            dropout: float=0.,
            kernel_size: int=3,
            num_layers: int=2,
            quantiles: list[int]=[0.1, 0.5, 0.9],
            lr: float=0.001,
            batch_size: int=32,
            num_epochs: int=10,
            verbose: bool=False,
            accelerator: str='auto',
            validation_size: float=0.,
        ):
        """
        Args:
            input_len: Length of an input time series (lookback window size).
            output_len: Length of an output time series (forecasting horizon).
            hidden_dim: Hidden dimensionality of TCN modules and the resnet-v module.
            dropout: Dropout rate.
            kernel_size: Kernel size of TCN modules.
            num_layers: Number of TCN modules.
            quantiles: List of predicted quantiles.
            lr: Learning rate.
            batch_size: Batch size.
            num_epochs: Number of epochs.
            verbose: Shows the progress bar during training.
            accelerator: Supports passing different accelerator types 
                ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as 
                custom accelerator instances.
            validation_size: The fraction (from 0 to 1) of train data for validation.
        """
        self.quantiles = quantiles
        super().__init__(
            input_len, output_len, hidden_dim, dropout, kernel_size, num_layers, 
            lr, batch_size, num_epochs, verbose, accelerator, validation_size
        )

    def _create_model(self):
        self.model = QuantileDeepTCNModule(
            target_dim=self.target_dim,
            past_cov_dim=self.past_cov_dim,
            future_cov_dim=self.future_cov_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            quantiles=self.quantiles,
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
            List[np.ndarray]: Quantile values represent distribution of the forecast. 
                All arrays are of the shape (n, m) where n is the output 
                length and m is the number of components. Order corresponds to
                quantiles in the initialization.
        """
        past_target, past_covariates, future_covariates = self._preprocess_input(
            past_target, past_covariates, future_covariates)
        with torch.no_grad():
            qvalues = self.model(past_target, past_covariates, future_covariates)
        qvalues = qvalues[0, -self.output_len:, :]
        qvalues = qvalues.cpu().numpy()
        return [qvalues[:, :, i] for i in range(len(self.quantiles))]

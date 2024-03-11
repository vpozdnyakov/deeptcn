from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from deeptcn.tcn import TCNModule
from deeptcn.data import SlidingWindowDataset


class ResNetVModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_in, h_in):
        # batch_size, seq_len, input_dim
        z, h = z_in, h_in
        batch_size, seq_len, input_dim = z.shape

        z = z.view(-1, input_dim)
        z = self.mlp(z)
        z = z.view(batch_size, seq_len, self.output_dim)
        output = self.dropout(z + h)

        return output


class DeepTCNModule(LightningModule):
    def __init__(
        self,
        target_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
        num_layers: int,
        lr: float,
        with_past_covariates: bool,
        with_future_covariates: bool,
    ):

        super().__init__()

        self.target_dim = target_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.lr = lr
        self.with_future_covariates = with_future_covariates
        self.with_past_covariates = with_past_covariates
        
        self.encoder = TCNModule(
            input_size=target_dim + past_cov_dim,
            target_size=hidden_dim,
            kernel_size=kernel_size,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            dropout=dropout,
            weight_norm=False,
        )

        if future_cov_dim:
            self.decoder = ResNetVModule(
                input_dim=future_cov_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
        self._create_output()

    def _create_output(self):
        pass

    def hidden_state(self, past_target, past_cov=None, future_cov=None):
        if self.with_past_covariates:
            past_target = torch.cat([past_target, past_cov], dim=2)
        h = self.encoder(past_target)
        if self.with_future_covariates:
            h = self.decoder(future_cov, h)
        return h

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class DeepTCN():
    def __init__(
            self, 
            input_len, 
            output_len, 
            hidden_dim, 
            dropout, 
            kernel_size, 
            num_layers, 
            lr, 
            batch_size, 
            num_epochs, 
            verbose,
            accelerator,
        ):

        assert output_len < input_len, "DeepTCN requires the output is stronly less than the input"

        self.input_len = input_len
        self.output_len = output_len

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.accelerator = accelerator

        self.with_past_covariates = None
        self.with_future_covariates = None
        self.past_cov_dim = None
        self.future_cov_dim = None
        self.target_dim = None
    
    def _create_model(self):
        pass

    def fit(self, target, past_covariates=None, future_covariates=None):
        self.with_past_covariates = True if past_covariates is not None else False
        self.with_future_covariates = True if future_covariates is not None else False
        self.past_cov_dim = past_covariates.shape[1] if self.with_past_covariates else 0
        self.future_cov_dim = future_covariates.shape[1] if self.with_future_covariates else 0
        self.target_dim = target.shape[1]
        
        self._create_model()
        self.dataset = SlidingWindowDataset(
            target=target,
            past_cov=past_covariates,
            future_cov=future_covariates, 
            window_size=self.input_len,
            step_size=1,
            shift_size=self.output_len,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.trainer = Trainer(
            enable_progress_bar=self.verbose,
            max_epochs=self.num_epochs,
            log_every_n_steps=np.ceil(len(self.dataloader) * 0.1),
            logger=CSVLogger('.'),
            accelerator=self.accelerator,
        )
        self.trainer.fit(
            model=self.model, 
            train_dataloaders=self.dataloader,
        )
        self.model.eval()
    
    def _preprocess_input(self, past_target, past_covariates, future_covariates):
        assert self.with_past_covariates == (past_covariates is not None), "Unexpected past covariates"
        assert self.with_future_covariates == (future_covariates is not None), "Unexpected future covariates"
        assert past_target.shape == (self.input_len, self.target_dim), "Unexpected input target shape"
        past_target = torch.as_tensor(
            past_target, dtype=torch.float32, device=self.model.device
            )[None, ...]
        if self.with_past_covariates:
            assert past_covariates.shape == (self.input_len, self.past_cov_dim), "Unexpected past covariance shape"
            past_covariates = torch.as_tensor(
                past_covariates, dtype=torch.float32, device=self.model.device
            )[None, ...]
        if self.with_future_covariates:
            assert future_covariates.shape == (self.output_len, self.future_cov_dim), "Unexpected future covariance shape"
            future_covariates = torch.as_tensor(
                future_covariates, dtype=torch.float32, device=self.model.device
            )[None, ...]
            future_covariates = F.pad(future_covariates, (0, 0, self.input_len - self.output_len, 0))
        return past_target, past_covariates, future_covariates

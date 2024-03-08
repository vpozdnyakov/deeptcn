import pandas as pd
import numpy as np
from deeptcn import QuantileDeepTCN

target = pd.DataFrame(
    np.sin(np.linspace(0, 100, 1000)) + np.random.randn(1000)*0.05, 
    columns=['target'],
)

input_len = 100 
output_len = 90

model = QuantileDeepTCN(
    input_len=input_len, 
    output_len=output_len,
    hidden_dim=4,
    num_layers=2,
    num_epochs=1,
    batch_size=16,
    lr=0.001,
    verbose=True,
)

model.fit(target)

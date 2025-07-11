# config.py
import torch

class Config:
    data_root = "data"
    batch_size = 32
    lr = 1e-4
    epochs = 50
    dct_block_size = 8
    num_dct_coeff = 15
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    weight_decay = 1e-4
    dropout_rate = 0.3
    patience = 5
    use_amp = True

import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

@dataclass
class FLClient:
    client_name: str
    data_loader: torch.utils.data.DataLoader

@dataclass
class FLMessage:
    client_name: str
    num_samples: int
    lora_sd: dict[str, torch.Tensor]

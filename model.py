from typing import override

import torch
from jaxtyping import Float
from torch import nn

hidden_layer_size = 10


class JointPolicy(nn.Module):
    def __init__(self):
        super(JointPolicy, self).__init__()
        self.fc1 = nn.Linear(4, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 1)
        self.relu = nn.ReLU()

    @override
    def forward(
        self,
        state: Float[torch.Tensor, "batch, 2"],
        target_state: Float[torch.Tensor, "batch, 2"],
    ) -> Float[torch.Tensor, "batch, 1"]:
        delta = target_state - state
        input = torch.cat([delta, state], dim=1)
        result = self.fc1(input)
        result = self.relu(result)
        result = self.fc2(result)
        result = self.relu(result)
        result = self.fc3(result)
        #result = self.relu(result)
        return result

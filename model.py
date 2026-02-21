import torch
from jaxtyping import Float
from torch import nn

hidden_layer_size = 10

class JointPolicy(nn.Module):
    def __init__(self, batch_size: int, num_actuators: int = 1):
        super(JointPolicy, self).__init__()
        self.batch_size = batch_size
        if num_actuators != 1:
            raise ValueError("Only num_actuators=1 is supported for now")
        self.num_actuators = 1

        self.layer_1 = nn.Linear(2, hidden_layer_size)
        self.layer_2 = nn.Linear(hidden_layer_size, 1)


    def forward(
        self,
        state: Float[torch.Tensor, "batch, 2"],
        target_state: Float[torch.Tensor, "batch, 2"],
    ) -> Float[torch.Tensor, "batch, 1"]:

        delta = target_state - state
        layer_1 = self.layer_1(delta)
        layer_2 = self.layer_2(layer_1)
        return layer_2

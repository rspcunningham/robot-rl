import torch
from jaxtyping import Float
from torch import nn

class JointPolicy(nn.Module):
    def __init__(self, batch_size: int):
        super(JointPolicy, self).__init__()
        self.batch_size = batch_size

        self.p_theta = nn.Parameter(torch.rand(())) * 10
        self.p_theta_dot = nn.Parameter(torch.rand(())) * 10


    def forward(self, state: Float[torch.Tensor, "batch, 2"], target_state: Float[torch.Tensor, "batch, 2"]):
        delta = target_state - state
        a = self.p_theta.expand(delta.shape[0])
        b = self.p_theta_dot.expand(delta.shape[0])
        return a + b

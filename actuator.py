import torch
from jaxtyping import Float

def get_applied_torque(
    torque_command: Float[torch.Tensor, "batch"],
    state: Float[torch.Tensor, "batch, 2"],
    damping_coefficient: float,
    max_applied_torque: float
) -> Float[torch.Tensor, "batch"]:

    clamped_torque_command = torch.clamp(torque_command, -max_applied_torque, max_applied_torque)
    damping_torque = damping_coefficient * state[:, 1]
    return clamped_torque_command - damping_torque

class BatchedActuator:
    max_applied_torque: float
    damping_coefficient: float
    batch_size: int
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        damping_coefficient: float,
        max_applied_torque: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype):

        self.max_applied_torque = max_applied_torque
        self.damping_coefficient = damping_coefficient
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def get_applied_torque(self, torque_command: Float[torch.Tensor, "batch"], joint_state: Float[torch.Tensor, "batch, 2"]) -> Float[torch.Tensor, "batch"]:
        return get_applied_torque(torque_command, joint_state, self.damping_coefficient, self.max_applied_torque)

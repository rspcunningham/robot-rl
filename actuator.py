import torch
from jaxtyping import Float

def get_applied_torque(
    torque_command: Float[torch.Tensor, "batch, 1"],
    state: Float[torch.Tensor, "batch, 2"],
    damping_coefficient: float,
    max_applied_torque: float
) -> Float[torch.Tensor, "batch, 1"]:

    if torque_command.ndim != 2 or torque_command.shape[1] != 1:
        raise ValueError("Expected torque_command with shape [batch, 1]")
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("Expected state with shape [batch, 2]")
    if state.shape[0] != torque_command.shape[0]:
        raise ValueError("Expected matching batch dimension between state and torque_command")
    clamped_torque_command = torch.clamp(torque_command, -max_applied_torque, max_applied_torque)
    damping_torque = damping_coefficient * state[:, 1:2]
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

    def get_applied_torque(
        self,
        torque_command: Float[torch.Tensor, "batch, 1"],
        joint_state: Float[torch.Tensor, "batch, 2"],
    ) -> Float[torch.Tensor, "batch, 1"]:
        return get_applied_torque(torque_command, joint_state, self.damping_coefficient, self.max_applied_torque)

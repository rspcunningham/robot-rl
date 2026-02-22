import torch
from jaxtyping import Float

from actuator import BatchedActuator

GRAVITY = 9.81
TIMESTEP_SIZE = 0.01

def get_theta_dot_dot(
    theta: Float[torch.Tensor, "batch, 1"],
    applied_torque: Float[torch.Tensor, "batch, 1"],
    mass: float,
    length: float,
) -> Float[torch.Tensor, "batch, 1"]:
    return (3 / mass / length**2) * (applied_torque - 0.5 * mass * GRAVITY * length * torch.cos(theta))

def step_dynamics(
    state: Float[torch.Tensor, "batch, 2"],
    applied_torque: Float[torch.Tensor, "batch, 1"],
    mass: float,
    length: float,
) -> Float[torch.Tensor, "batch, 2"]:

    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("Expected state with shape [batch, 2]")
    if applied_torque.ndim != 2 or applied_torque.shape[1] != 1:
        raise ValueError("Expected applied_torque with shape [batch, 1]")
    if state.shape[0] != applied_torque.shape[0]:
        raise ValueError("Expected matching batch dimension between state and applied_torque")

    theta_dot_dot = get_theta_dot_dot(state[:, 0:1], applied_torque, mass, length)
    new_theta_dot = state[:, 1:2] + theta_dot_dot * TIMESTEP_SIZE
    new_theta = state[:, 0:1] + new_theta_dot * TIMESTEP_SIZE
    return torch.cat([new_theta, new_theta_dot], dim=1)

class BatchedJoint:
    mass: float
    length: float
    state: Float[torch.Tensor, "batch, 2"]
    actuator: BatchedActuator | None
    batch_size: int
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        mass: float,
        length: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        actuator: BatchedActuator | None = None,
    ):
        self.mass = mass
        self.length = length
        self.actuator = actuator
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.reset()

    def step_dynamics(
        self,
        torque_command: Float[torch.Tensor, "batch, 1"],
    ) -> Float[torch.Tensor, "batch, 1"]:
        if torque_command.ndim != 2 or torque_command.shape[1] != 1:
            raise ValueError("Expected torque_command with shape [batch, 1]")
        if torque_command.shape[0] != self.batch_size:
            raise ValueError("Expected torque_command batch dimension to match system batch_size")

        applied_torque = torque_command
        if self.actuator is not None:
            applied_torque = self.actuator.get_applied_torque(torque_command, self.state)
        self.state = step_dynamics(self.state, applied_torque, self.mass, self.length)
        return applied_torque

    def reset(self) -> None:
        theta = torch.rand((self.batch_size, 1), device=self.device, dtype=self.dtype) * 2 * torch.pi
        theta_dot = torch.zeros_like(theta)

        self.state = torch.concat([theta, theta_dot], dim=1)

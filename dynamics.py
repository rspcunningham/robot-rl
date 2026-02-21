import torch
from jaxtyping import Float

from actuator import BatchedActuator

GRAVITY = 9.81
TIMESTEP_SIZE = 0.01

def get_theta_dot_dot(
    theta: Float[torch.Tensor, "batch"],
    applied_torque: Float[torch.Tensor, "batch"],
    mass: float,
    length: float,
) -> Float[torch.Tensor, "batch"]:
    return (3 / mass / length**2) * (applied_torque - 0.5 * mass * GRAVITY * length * torch.cos(theta))

def step_dynamics(
    state: Float[torch.Tensor, "batch, 2"],
    applied_torque: Float[torch.Tensor, "batch"],
    mass: float,
    length: float,
) -> Float[torch.Tensor, "batch, 2"]:

    theta_dot_dot = get_theta_dot_dot(state[:, 0], applied_torque, mass, length)
    state[:, 1] += theta_dot_dot * TIMESTEP_SIZE
    state[:, 0] += state[:, 1] * TIMESTEP_SIZE
    return state

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

    def step_dynamics(self, torque_command: Float[torch.Tensor, "batch"]) -> Float[torch.Tensor, "batch"]:
        applied_torque = torque_command
        if self.actuator is not None:
            applied_torque = self.actuator.get_applied_torque(torque_command, self.state)
        self.state = step_dynamics(self.state, applied_torque, self.mass, self.length)
        return applied_torque

    def reset(self) -> None:
        self.state = torch.zeros(self.batch_size, 2, device=self.device, dtype=self.dtype)

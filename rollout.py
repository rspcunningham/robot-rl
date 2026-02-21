import torch

from actuator import BatchedActuator
from dynamics import BatchedJoint, TIMESTEP_SIZE
from model import JointPolicy
from util import plot_rollout_traces

device = torch.device("cpu")
dtype = torch.float32

mass = 1
length = 1
batch_size = 1
sim_length = 1000
damping_coeff = 1
max_applied_torque = 10

target_state = torch.zeros(batch_size, 2, device=device, dtype=dtype)

actuator = BatchedActuator(
    damping_coefficient=damping_coeff,
    max_applied_torque=max_applied_torque,
    batch_size=batch_size,
    device=device,
    dtype=dtype,
)

system = BatchedJoint(mass, length, batch_size, device, dtype, actuator=actuator)
policy = JointPolicy(batch_size)

state_trace = torch.zeros(batch_size, sim_length + 1, 2, device=device, dtype=dtype)
tau_trace = torch.zeros(batch_size, sim_length + 1, device=device, dtype=dtype)

state_trace[:, 0, :] = system.state.clone()
tau_trace[:, 0] = torch.zeros(batch_size, device=device, dtype=dtype)

for i in range(sim_length):
    torque_command = policy(state_trace[:, i, :], target_state)
    eff_applied_torque = system.step_dynamics(torque_command)
    state_trace[:, i + 1, :] = system.state.clone()
    tau_trace[:, i + 1] = eff_applied_torque.clone()

state_trace = state_trace.detach().cpu()
tau_trace = tau_trace.detach().cpu()

plot_rollout_traces(state_trace, tau_trace, TIMESTEP_SIZE)

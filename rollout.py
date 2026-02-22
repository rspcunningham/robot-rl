from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import torch

from actuator import BatchedActuator
from dynamics import TIMESTEP_SIZE, BatchedJoint
from model import JointPolicy
from util import plot_rollout_traces

DEVICE = torch.device("cpu")
DTYPE = torch.float32
RolloutMode = Literal["state", "error"]

MASS = 1
LENGTH = 1
BATCH_SIZE = 1
SIM_LENGTH = 1000
DAMPING_COEFF = 1
MAX_APPLIED_TORQUE = 10


def _load_policy_from_checkpoint(policy: JointPolicy, checkpoint_path: Path) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        step = None
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    policy.load_state_dict(state_dict)
    return step


def _to_batch_state(theta: float, theta_dot: float, batch_size: int) -> torch.Tensor:
    return torch.tensor([[theta, theta_dot]], device=DEVICE, dtype=DTYPE).repeat(batch_size, 1)


def run_rollout(
    checkpoint_path: str | Path,
    output_path: str | Path,
    target_state: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    sim_length: int = SIM_LENGTH,
    batch_size: int = BATCH_SIZE,
    mode: RolloutMode = "error",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    actuator = BatchedActuator(
        damping_coefficient=DAMPING_COEFF,
        max_applied_torque=MAX_APPLIED_TORQUE,
        batch_size=batch_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    system = BatchedJoint(MASS, LENGTH, batch_size, DEVICE, DTYPE, actuator=actuator)
    policy = JointPolicy()
    loaded_step = _load_policy_from_checkpoint(policy, checkpoint_path)
    policy.eval()

    if target_state is None:
        target_state = torch.zeros(batch_size, 2, device=DEVICE, dtype=DTYPE)
    else:
        target_state = target_state.to(device=DEVICE, dtype=DTYPE)
        if target_state.ndim != 2 or target_state.shape[1] != 2:
            raise ValueError("Expected target_state shape [batch, 2] or [1, 2]")
        if target_state.shape[0] == 1 and batch_size > 1:
            target_state = target_state.repeat(batch_size, 1)
        elif target_state.shape[0] != batch_size:
            raise ValueError("Expected target_state batch dimension to match batch_size")

    if initial_state is not None:
        initial_state = initial_state.to(device=DEVICE, dtype=DTYPE)
        if initial_state.ndim != 2 or initial_state.shape[1] != 2:
            raise ValueError("Expected initial_state shape [batch, 2] or [1, 2]")
        if initial_state.shape[0] == 1 and batch_size > 1:
            initial_state = initial_state.repeat(batch_size, 1)
        elif initial_state.shape[0] != batch_size:
            raise ValueError("Expected initial_state batch dimension to match batch_size")
        system.state = initial_state.clone()

    state_trace = torch.zeros(batch_size, sim_length + 1, 2, device=DEVICE, dtype=DTYPE)
    tau_trace = torch.zeros(batch_size, sim_length + 1, 1, device=DEVICE, dtype=DTYPE)

    state_trace[:, 0, :] = system.state.clone()
    tau_trace[:, 0, :] = torch.zeros(batch_size, 1, device=DEVICE, dtype=DTYPE)

    for i in range(sim_length):
        with torch.no_grad():
            torque_command = policy(state_trace[:, i, :], target_state)

        eff_applied_torque = system.step_dynamics(torque_command)
        state_trace[:, i + 1, :] = system.state.clone()
        tau_trace[:, i + 1, :] = eff_applied_torque.clone()

    state_trace = state_trace.detach().cpu()
    tau_trace = tau_trace.detach().cpu()
    target_state_cpu = target_state.detach().cpu()

    plot_rollout_traces(
        state_trace=state_trace,
        tau_trace=tau_trace,
        timestep_size=TIMESTEP_SIZE,
        output_path=str(output_path),
        target_state=target_state_cpu,
        mode=mode,
    )
    if loaded_step is not None:
        print(f"Completed rollout for step {loaded_step} from {checkpoint_path}")
    else:
        print(f"Completed rollout from {checkpoint_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one rollout for a checkpoint and write a plot.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output", default="rollout_traces.png", help="Output image path.")
    parser.add_argument("--sim-length", type=int, default=SIM_LENGTH, help="Rollout length in steps.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for rollout simulation.")
    parser.add_argument("--mode", choices=("state", "error"), default="error", help="Rollout plot mode.")
    parser.add_argument("--target-theta", type=float, default=0.0, help="Target theta in radians.")
    parser.add_argument("--target-theta-dot", type=float, default=0.0, help="Target angular velocity in rad/s.")
    parser.add_argument("--init-theta", type=float, default=0.0, help="Initial theta in radians.")
    parser.add_argument("--init-theta-dot", type=float, default=0.0, help="Initial angular velocity in rad/s.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = cast(RolloutMode, args.mode)
    target_state = _to_batch_state(args.target_theta, args.target_theta_dot, BATCH_SIZE)
    initial_state = _to_batch_state(args.init_theta, args.init_theta_dot, BATCH_SIZE)
    run_rollout(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        target_state=target_state,
        initial_state=initial_state,
        sim_length=args.sim_length,
        batch_size=args.batch_size,
        mode=mode,
    )


if __name__ == "__main__":
    main()

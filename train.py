from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from actuator import BatchedActuator
from dynamics import BatchedJoint
from model import JointPolicy
from rollout import run_rollout
from util import plot_loss_curves

DEVICE = torch.device("cpu")
DTYPE = torch.float32

MASS = 1
LENGTH = 1
DAMPING_COEFF = 1
MAX_APPLIED_TORQUE = 10

SIM_LENGTH = 1000
BATCH_SIZE = 100
TRAINING_STEPS = 100

EVAL_EVERY = 20
CHECKPOINT_EVERY = 20

ROLLOUT_SIM_LENGTH = 1000
ROLLOUT_BATCH_SIZE = BATCH_SIZE
ROLLOUT_TARGET_THETA = 0.0
ROLLOUT_TARGET_THETA_DOT = 0.0
ROLLOUT_SEED = 0


def get_target_state(batch_size: int) -> torch.Tensor:
    target_theta = torch.rand(batch_size, 1, device=DEVICE, dtype=DTYPE) * 2 * torch.pi
    target_theta_dot = torch.zeros_like(target_theta)
    return torch.concat([target_theta, target_theta_dot], dim=1)


def get_loss(target_state: torch.Tensor, final_state: torch.Tensor) -> torch.Tensor:
    theta_loss_weight = 1
    theta_dot_loss_weight = 5

    individual = torch.abs(target_state - final_state) ** 2
    theta_loss = individual[:, 0]
    theta_dot_loss = torch.abs(target_state[:, 1] - final_state[:, 1]) ** 2
    return theta_loss_weight * theta_loss.mean() + theta_dot_loss_weight * theta_dot_loss.mean()


def eval_policy(
    policy: JointPolicy,
    system: BatchedJoint,
    target_state: torch.Tensor,
    sim_length: int,
) -> torch.Tensor:
    system.reset()
    state_trace = torch.zeros((SIM_LENGTH + 1, BATCH_SIZE, 2), device=DEVICE, dtype=DTYPE)
    state_trace[0] = system.state

    with torch.no_grad():
        for t in range(sim_length):
            torque_command = policy(system.state, target_state)
            _ = system.step_dynamics(torque_command)
            state_trace[t + 1] = system.state
        loss = get_loss(target_state, state_trace)
    return loss


def save_metrics_csv(metrics: dict[str, list[dict[str, float | int]]], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "step", "loss"])
        writer.writeheader()
        for row in metrics["train_losses"]:
            writer.writerow({"split": "train", "step": int(row["step"]), "loss": float(row["loss"])})
        for row in metrics["eval_losses"]:
            writer.writerow({"split": "eval", "step": int(row["step"]), "loss": float(row["loss"])})
    print(f"Saved metrics to {metrics_path}")


def save_checkpoint(policy: JointPolicy, checkpoint_path: Path, step: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": policy.state_dict(),
        },
        checkpoint_path,
    )



run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / run_timestamp
checkpoints_dir = run_dir / "checkpoints"
rollouts_dir = run_dir / "rollouts"
metrics_path = run_dir / "metrics.csv"
losses_plot_path = run_dir / "loss_curves.png"

actuator = BatchedActuator(
    damping_coefficient=DAMPING_COEFF,
    max_applied_torque=MAX_APPLIED_TORQUE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    dtype=DTYPE,
)
system = BatchedJoint(MASS, LENGTH, BATCH_SIZE, DEVICE, DTYPE, actuator=actuator)
policy = JointPolicy()
optim = torch.optim.Adam(policy.parameters(), lr=0.01)

metrics: dict[str, list[dict[str, float | int]]] = {
    "train_losses": [],
    "eval_losses": [],
}

for step in tqdm(range(TRAINING_STEPS)):
    target_state = get_target_state(BATCH_SIZE)
    state_trace = torch.zeros((SIM_LENGTH + 1, BATCH_SIZE, 2), device=DEVICE, dtype=DTYPE)
    system.reset()

    state_trace[0] = system.state

    for t in range(SIM_LENGTH):
        torque_command = policy(system.state, target_state)
        _ = system.step_dynamics(torque_command)
        state_trace[t + 1] = system.state

    loss = get_loss(target_state, state_trace)
    optim.zero_grad()
    loss.backward()
    optim.step()

    metrics["train_losses"].append({"step": step, "loss": float(loss.item())})

    if step % EVAL_EVERY == 0:
        eval_target_state = get_target_state(BATCH_SIZE)
        eval_loss = eval_policy(policy=policy, system=system, target_state=eval_target_state, sim_length=SIM_LENGTH)
        print(f"Eval loss at step {step}: {eval_loss.item():.6f}")
        metrics["eval_losses"].append({"step": step, "loss": float(eval_loss.item())})

    if step % CHECKPOINT_EVERY == 0:
        checkpoint_path = checkpoints_dir / f"ch_{step:06d}.pt"
        save_checkpoint(policy, checkpoint_path, step=step)

if (TRAINING_STEPS - 1) % CHECKPOINT_EVERY != 0:
    final_step = TRAINING_STEPS - 1
    checkpoint_path = checkpoints_dir / f"ch_{final_step:06d}.pt"
    save_checkpoint(policy, checkpoint_path, step=final_step)

save_metrics_csv(metrics, metrics_path)
plot_loss_curves(metrics["train_losses"], metrics["eval_losses"], output_path=str(losses_plot_path))

rollout_generator = torch.Generator(device=DEVICE)
_ = rollout_generator.manual_seed(ROLLOUT_SEED)
rollout_initial_theta = torch.rand(
    (ROLLOUT_BATCH_SIZE, 1),
    device=DEVICE,
    dtype=DTYPE,
    generator=rollout_generator,
) * 2 * torch.pi
rollout_initial_theta_dot = torch.zeros_like(rollout_initial_theta)
rollout_initial_state = torch.concat([rollout_initial_theta, rollout_initial_theta_dot], dim=1)
rollout_target_state = torch.tensor(
    [[ROLLOUT_TARGET_THETA, ROLLOUT_TARGET_THETA_DOT]],
    device=DEVICE,
    dtype=DTYPE,
).repeat(ROLLOUT_BATCH_SIZE, 1)

checkpoint_paths = sorted(checkpoints_dir.glob("ch_*.pt"))
for checkpoint_path in checkpoint_paths:
    rollout_path = rollouts_dir / f"{checkpoint_path.stem}.png"
    _ = run_rollout(
        checkpoint_path=checkpoint_path,
        output_path=rollout_path,
        target_state=rollout_target_state,
        initial_state=rollout_initial_state,
        sim_length=ROLLOUT_SIM_LENGTH,
        batch_size=ROLLOUT_BATCH_SIZE,
        mode="error",
    )

print(f"Training run complete: {run_dir}")

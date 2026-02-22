import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FuncFormatter, MultipleLocator

LINE_COLORS = ("#ff6b6b", "#ffd166", "#f4a261", "#e76f51", "#c77dff", "#f2cc8f")


def _format_pi_ticks(value: float, _pos: int) -> str:
    quarter_turns = round((value / math.pi) * 4)
    snapped_value = (quarter_turns / 4) * math.pi
    if not math.isclose(value, snapped_value, rel_tol=1e-6, abs_tol=1e-6):
        return f"{value:.2f}"

    if quarter_turns == 0:
        return "0"

    sign = "-" if quarter_turns < 0 else ""
    numerator = abs(quarter_turns)
    denominator = 4
    gcd = math.gcd(numerator, denominator)
    numerator //= gcd
    denominator //= gcd

    if denominator == 1:
        if numerator == 1:
            return f"{sign}π"
        return f"{sign}{numerator}π"
    if numerator == 1:
        return f"{sign}π/{denominator}"
    return f"{sign}{numerator}π/{denominator}"


def _normalize_target_state(target_state: torch.Tensor, batch_size: int) -> torch.Tensor:
    target_state_for_plot = target_state.detach().cpu()
    if target_state_for_plot.ndim == 1:
        if target_state_for_plot.shape[0] != 2:
            raise ValueError("Expected target_state shape [2], [batch, 2], or [1, 2]")
        target_state_for_plot = target_state_for_plot.unsqueeze(0)
    elif target_state_for_plot.ndim != 2 or target_state_for_plot.shape[1] != 2:
        raise ValueError("Expected target_state shape [2], [batch, 2], or [1, 2]")

    if target_state_for_plot.shape[0] == 1 and batch_size > 1:
        target_state_for_plot = target_state_for_plot.repeat(batch_size, 1)
    elif target_state_for_plot.shape[0] != batch_size:
        raise ValueError("Expected target_state batch dimension to match state_trace batch size")

    return target_state_for_plot


def plot_rollout_traces(
    state_trace: torch.Tensor,
    tau_trace: torch.Tensor,
    timestep_size: float,
    output_path: str = "rollout_traces.png",
    target_state: torch.Tensor | None = None,
    mode: Literal["state", "error"] = "state",
) -> None:
    if mode not in ("state", "error"):
        raise ValueError("Expected mode to be one of: 'state', 'error'")

    batch_size = state_trace.shape[0]
    sim_length = state_trace.shape[1] - 1
    time_trace = torch.arange(sim_length + 1, device=state_trace.device, dtype=state_trace.dtype) * timestep_size
    time_trace_cpu = time_trace.cpu()

    target_state_for_plot = None
    if target_state is not None:
        target_state_for_plot = _normalize_target_state(target_state, batch_size)

    with plt.style.context("dark_background"):
        if mode == "state":
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
            fig.patch.set_facecolor("#111418")
            labels = ("θ [rad]", "dθ/dt [rad/s]", "τ [N·m]")
            state_traces = (state_trace[:, :, 0], state_trace[:, :, 1])

            for axis in axes:
                axis.set_facecolor("#171b21")
                axis.tick_params(colors="#dce3ea")
                axis.set_prop_cycle(color=LINE_COLORS)

            for state_idx, (axis, trace, ylabel) in enumerate(zip(axes[:2], state_traces, labels[:2])):
                for batch_idx in range(batch_size):
                    color = LINE_COLORS[batch_idx % len(LINE_COLORS)]
                    line_label = f"batch {batch_idx}" if batch_size > 1 else None
                    axis.plot(
                        time_trace_cpu,
                        trace[batch_idx].cpu(),
                        label=line_label,
                        linewidth=1.8,
                        color=color,
                    )
                    if target_state_for_plot is not None:
                        target_label = f"target batch {batch_idx}" if batch_size > 1 else "target"
                        axis.plot(
                            time_trace_cpu,
                            torch.full_like(time_trace_cpu, target_state_for_plot[batch_idx, state_idx].item()),
                            linestyle="--",
                            linewidth=1.6,
                            color=color,
                            alpha=0.9,
                            label=target_label,
                        )
                axis.set_ylabel(ylabel, color="#dce3ea")
                axis.yaxis.set_major_locator(MultipleLocator(base=math.pi / 4))
                axis.yaxis.set_major_formatter(FuncFormatter(_format_pi_ticks))
                axis.grid(True, color="#4b5563", alpha=0.35)
                if batch_size > 1 or target_state_for_plot is not None:
                    legend = axis.legend(loc="upper right")
                    legend.get_frame().set_facecolor("#111418")
                    legend.get_frame().set_edgecolor("#4b5563")

            tau_axis = axes[2]
            if tau_trace.ndim == 2:
                for batch_idx in range(batch_size):
                    line_label = f"batch {batch_idx}" if batch_size > 1 else None
                    tau_axis.plot(time_trace_cpu, tau_trace[batch_idx].cpu(), label=line_label, linewidth=1.8)
            elif tau_trace.ndim == 3:
                num_actuators = tau_trace.shape[2]
                for batch_idx in range(batch_size):
                    for actuator_idx in range(num_actuators):
                        line_label = None
                        if batch_size > 1 or num_actuators > 1:
                            line_label = f"batch {batch_idx}, actuator {actuator_idx}"
                        tau_axis.plot(
                            time_trace_cpu,
                            tau_trace[batch_idx, :, actuator_idx].cpu(),
                            label=line_label,
                            linewidth=1.8,
                        )
            else:
                raise ValueError("Expected tau_trace shape [batch, time] or [batch, time, actuators]")

            tau_axis.set_ylabel(labels[2], color="#dce3ea")
            tau_axis.grid(True, color="#4b5563", alpha=0.35)
            if tau_trace.ndim == 2:
                if batch_size > 1:
                    legend = tau_axis.legend(loc="upper right")
                    legend.get_frame().set_facecolor("#111418")
                    legend.get_frame().set_edgecolor("#4b5563")
            elif batch_size > 1 or tau_trace.shape[2] > 1:
                legend = tau_axis.legend(loc="upper right")
                legend.get_frame().set_facecolor("#111418")
                legend.get_frame().set_edgecolor("#4b5563")

            axes[-1].set_xlabel("time [s]", color="#dce3ea")
            fig.suptitle("Rollout traces", color="#f3f4f6")
            fig.tight_layout()

        else:
            if target_state_for_plot is None:
                raise ValueError("target_state is required when mode='error'")

            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
            fig.patch.set_facecolor("#111418")

            for axis in axes:
                axis.set_facecolor("#171b21")
                axis.tick_params(colors="#dce3ea")

            state_error_trace = target_state_for_plot[:, None, :] - state_trace.detach().cpu()
            error_labels = ("θ error [rad]", "dθ/dt error [rad/s]")
            state_names = ("θ", "dθ/dt")

            for state_idx, (axis, ylabel) in enumerate(zip(axes, error_labels)):
                color = LINE_COLORS[state_idx % len(LINE_COLORS)]
                error_dim_trace = state_error_trace[:, :, state_idx]
                mean_error = error_dim_trace.mean(dim=0)
                min_error = error_dim_trace.min(dim=0).values
                max_error = error_dim_trace.max(dim=0).values

                if batch_size > 1:
                    axis.fill_between(
                        time_trace_cpu,
                        min_error,
                        max_error,
                        color=color,
                        alpha=0.22,
                        label="spread (min/max)",
                    )

                axis.plot(
                    time_trace_cpu,
                    mean_error,
                    color=color,
                    linewidth=2.3,
                    label="mean error",
                )
                axis.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.2, alpha=0.8, label="zero error")
                axis.set_ylabel(ylabel, color="#dce3ea")
                axis.grid(True, color="#4b5563", alpha=0.35)

                if state_idx == 0:
                    axis.yaxis.set_major_locator(MultipleLocator(base=math.pi / 4))
                    axis.yaxis.set_major_formatter(FuncFormatter(_format_pi_ticks))

                legend = axis.legend(loc="upper right", title=state_names[state_idx])
                legend.get_frame().set_facecolor("#111418")
                legend.get_frame().set_edgecolor("#4b5563")

            axes[-1].set_xlabel("time [s]", color="#dce3ea")
            fig.suptitle("Target Tracking Error (target - state)", color="#f3f4f6")
            fig.tight_layout()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)
    print(f"Saved rollout plot to {output_path_obj}")


def plot_loss_curves(
    train_losses: list[dict[str, float | int]],
    eval_losses: list[dict[str, float | int]],
    output_path: str = "loss_curves.png",
) -> None:
    if not train_losses and not eval_losses:
        raise ValueError("No train or eval losses provided")

    train_steps = [int(point["step"]) for point in train_losses]
    train_values = [float(point["loss"]) for point in train_losses]
    eval_steps = [int(point["step"]) for point in eval_losses]
    eval_values = [float(point["loss"]) for point in eval_losses]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.patch.set_facecolor("#111418")
        ax.set_facecolor("#171b21")
        ax.tick_params(colors="#dce3ea")

        if train_steps:
            ax.plot(train_steps, train_values, linewidth=1.8, color="#ff6b6b", label="train")
        if eval_steps:
            ax.plot(eval_steps, eval_values, linewidth=2.2, color="#ffd166", marker="o", markersize=3, label="eval")

        ax.set_xlabel("step", color="#dce3ea")
        ax.set_ylabel("loss", color="#dce3ea")
        ax.set_title("Train / Eval Loss Over Time", color="#f3f4f6")
        ax.grid(True, color="#4b5563", alpha=0.35)
        legend = ax.legend(loc="upper right")
        legend.get_frame().set_facecolor("#111418")
        legend.get_frame().set_edgecolor("#4b5563")
        fig.tight_layout()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to {output_path_obj}")

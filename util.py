import math

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


def plot_rollout_traces(
    state_trace: torch.Tensor,
    tau_trace: torch.Tensor,
    timestep_size: float,
    output_path: str = "rollout_traces.png",
) -> None:
    batch_size = state_trace.shape[0]
    sim_length = state_trace.shape[1] - 1
    time_trace = torch.arange(sim_length + 1, device=state_trace.device, dtype=state_trace.dtype) * timestep_size

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        fig.patch.set_facecolor("#111418")
        labels = ("θ [rad]", "dθ/dt [rad/s]", "τ [N·m]")
        state_traces = (state_trace[:, :, 0], state_trace[:, :, 1])

        for axis in axes:
            axis.set_facecolor("#171b21")
            axis.tick_params(colors="#dce3ea")
            axis.set_prop_cycle(color=LINE_COLORS)

        for axis, trace, ylabel in zip(axes[:2], state_traces, labels[:2]):
            for batch_idx in range(batch_size):
                line_label = f"batch {batch_idx}" if batch_size > 1 else None
                axis.plot(time_trace.cpu(), trace[batch_idx].cpu(), label=line_label, linewidth=1.8)
            axis.set_ylabel(ylabel, color="#dce3ea")
            axis.yaxis.set_major_locator(MultipleLocator(base=math.pi / 4))
            axis.yaxis.set_major_formatter(FuncFormatter(_format_pi_ticks))
            axis.grid(True, color="#4b5563", alpha=0.35)
            if batch_size > 1:
                legend = axis.legend(loc="upper right")
                legend.get_frame().set_facecolor("#111418")
                legend.get_frame().set_edgecolor("#4b5563")

        tau_axis = axes[2]
        if tau_trace.ndim == 2:
            for batch_idx in range(batch_size):
                line_label = f"batch {batch_idx}" if batch_size > 1 else None
                tau_axis.plot(time_trace.cpu(), tau_trace[batch_idx].cpu(), label=line_label, linewidth=1.8)
        elif tau_trace.ndim == 3:
            num_actuators = tau_trace.shape[2]
            for batch_idx in range(batch_size):
                for actuator_idx in range(num_actuators):
                    line_label = None
                    if batch_size > 1 or num_actuators > 1:
                        line_label = f"batch {batch_idx}, actuator {actuator_idx}"
                    tau_axis.plot(
                        time_trace.cpu(),
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

    if "agg" in plt.get_backend().lower():
        fig.savefig(output_path, dpi=150)
        print(f"Saved rollout plot to {output_path}")
    else:
        plt.show()

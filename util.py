import matplotlib.pyplot as plt
import torch


def plot_rollout_traces(
    state_trace: torch.Tensor,
    tau_trace: torch.Tensor,
    timestep_size: float,
    output_path: str = "rollout_traces.png",
) -> None:
    batch_size = state_trace.shape[0]
    sim_length = state_trace.shape[1] - 1
    time_trace = torch.arange(sim_length + 1, device=state_trace.device, dtype=state_trace.dtype) * timestep_size

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = ("theta [rad]", "theta_dot [rad/s]", "tau [N*m]")
    traces = (state_trace[:, :, 0], state_trace[:, :, 1], tau_trace)

    for axis, trace, ylabel in zip(axes, traces, labels):
        for batch_idx in range(batch_size):
            line_label = f"batch {batch_idx}" if batch_size > 1 else None
            axis.plot(time_trace.cpu(), trace[batch_idx].cpu(), label=line_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if batch_size > 1:
            axis.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Rollout traces")
    fig.tight_layout()

    if "agg" in plt.get_backend().lower():
        fig.savefig(output_path, dpi=150)
        print(f"Saved rollout plot to {output_path}")
    else:
        plt.show()

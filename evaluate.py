"""
Evaluation — PD-only vs PD+RL comparison.

Generates comprehensive plots:
  1. Trajectory comparison (YZ-plane)
  2. Tracking error over time
  3. Per-axis tracking
  4. EE speed (smoothness verification)
  5. Error distribution histogram
  6. RL actions over time
  7. Zoomed speed (jitter check)
  8. Summary metrics bar chart
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from env import FrankaTrackingEnv
import os

os.system('cls')

def run_episode(env, model=None, n_steps=650 , seed=42):
    """Run a full episode, collect detailed data."""
    obs, _ = env.reset(seed=seed)
    data = {
        "time": [], "ee": [], "des": [], "err_mm": [],
        "reward": [], "actions": [],
    }
    for i in range(n_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape[0])

        obs, reward, terminated, truncated, info = env.step(action)

        data["time"].append(env.sim_time)
        data["ee"].append(info["ee_pos"].copy())
        data["des"].append(info["des_pos"].copy())
        data["err_mm"].append(info["tracking_error_mm"])
        data["reward"].append(reward)
        data["actions"].append(action.copy())

        if truncated:
            break

    for k in data:
        data[k] = np.array(data[k])
    return data


def compute_metrics(data, dt, skip=65):
    """Compute tracking and smoothness metrics."""
    e = data["err_mm"][skip:]
    ee = data["ee"][skip:]
    vel = np.gradient(ee, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1) * 1000
    jerk_mag = np.linalg.norm(jerk, axis=1)

    return {
        "mean_mm": np.mean(e),
        "max_mm": np.max(e),
        "std_mm": np.std(e),
        "rmse_mm": np.sqrt(np.mean(e**2)),
        "p95_mm": np.percentile(e, 95),
        "speed_std": np.std(speed),
        "mean_jerk": np.mean(jerk_mag),
    }


# ── Run both conditions ──
env = FrankaTrackingEnv(
    obs_noise_std=0.005,
    enable_delay=True,
    ctrl_substeps=10,
    residual_scale=0.005,
)
dt = env.dt_ctrl
skip = 65   

print("Running PD-only baseline...")
d_pd = run_episode(env, model=None)
m_pd = compute_metrics(d_pd, dt)

print("Running PD + RL...")
rl_model = PPO.load("models/ppo_best")
d_rl = run_episode(env, model=rl_model)
m_rl = compute_metrics(d_rl, dt)

# ── Print results ──
print(f"\n{'='*60}")
print(f"  FINAL RESULTS")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'PD Only':>12} {'PD + RL':>12} {'Change':>10}")
print(f"  {'-'*57}")
for key, label in [("mean_mm", "Mean error (mm)"),
                    ("max_mm", "Max error (mm)"),
                    ("std_mm", "Std error (mm)"),
                    ("rmse_mm", "RMSE (mm)"),
                    ("p95_mm", "95th percentile (mm)"),
                    ("speed_std", "Speed std (mm/s)"),
                    ("mean_jerk", "Mean jerk (m/s³)")]:
    v_pd = m_pd[key]
    v_rl = m_rl[key]
    pct = ((v_rl - v_pd) / v_pd) * 100
    print(f"  {label:<25} {v_pd:>10.3f}   {v_rl:>10.3f}   {pct:>+8.1f}%")
print(f"{'='*60}")

# ── Generate plots ──
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
fig.suptitle("Franka Panda End-Effector Tracking — Final Results\n"
             "(Circle 0.08Hz, r=200mm | Noise σ=5mm + 20ms Delay)",
             fontsize=14, fontweight="bold")

# 1. Trajectory YZ
ax = fig.add_subplot(gs[0, 0])
ax.plot(d_pd["des"][:,1]*1e3, d_pd["des"][:,2]*1e3, 'k--', lw=2, label="Desired", alpha=0.5)
ax.plot(d_pd["ee"][:,1]*1e3, d_pd["ee"][:,2]*1e3, 'C0-', lw=0.8, label="PD Only", alpha=0.6)
ax.plot(d_rl["ee"][:,1]*1e3, d_rl["ee"][:,2]*1e3, 'C3-', lw=0.8, label="PD+RL", alpha=0.8)
ax.set_xlabel("Y (mm)"); ax.set_ylabel("Z (mm)")
ax.set_title("Trajectory (YZ-Plane)")
ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# 2. Tracking error over time
ax = fig.add_subplot(gs[0, 1:])
ax.plot(d_pd["time"], d_pd["err_mm"], 'C0-', lw=0.6, label="PD Only", alpha=0.6)
ax.plot(d_rl["time"], d_rl["err_mm"], 'C3-', lw=0.6, label="PD+RL", alpha=0.7)
ax.axhline(3.0, color='gray', ls=':', alpha=0.5, label="3mm target")
ax.fill_between(d_pd["time"], 0, d_pd["err_mm"], alpha=0.06, color='C0')
ax.fill_between(d_rl["time"], 0, d_rl["err_mm"], alpha=0.06, color='C3')
ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (mm)")
ax.set_title("Tracking Error Over Time")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 3. Y-axis tracking
ax = fig.add_subplot(gs[1, 0])
ax.plot(d_pd["time"], d_pd["des"][:,1]*1e3, 'k--', lw=1.5, label="Desired", alpha=0.5)
ax.plot(d_pd["time"], d_pd["ee"][:,1]*1e3, 'C0-', lw=0.8, label="PD", alpha=0.6)
ax.plot(d_rl["time"], d_rl["ee"][:,1]*1e3, 'C3-', lw=0.8, label="PD+RL", alpha=0.8)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Y (mm)")
ax.set_title("Y-Axis Tracking")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 4. Z-axis tracking
ax = fig.add_subplot(gs[1, 1])
ax.plot(d_pd["time"], d_pd["des"][:,2]*1e3, 'k--', lw=1.5, label="Desired", alpha=0.5)
ax.plot(d_pd["time"], d_pd["ee"][:,2]*1e3, 'C0-', lw=0.8, label="PD", alpha=0.6)
ax.plot(d_rl["time"], d_rl["ee"][:,2]*1e3, 'C3-', lw=0.8, label="PD+RL", alpha=0.8)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Z (mm)")
ax.set_title("Z-Axis Tracking")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 5. Speed (smoothness)
vel_pd = np.gradient(d_pd["ee"], dt, axis=0)
vel_rl = np.gradient(d_rl["ee"], dt, axis=0)
spd_pd = np.linalg.norm(vel_pd, axis=1) * 1000
spd_rl = np.linalg.norm(vel_rl, axis=1) * 1000
ax = fig.add_subplot(gs[1, 2])
ax.plot(d_pd["time"][skip:], spd_pd[skip:], 'C0-', lw=0.6, label="PD", alpha=0.5)
ax.plot(d_rl["time"][skip:], spd_rl[skip:], 'C3-', lw=0.6, label="PD+RL", alpha=0.7)

# Ideal speed for 0.08Hz circle: 2*pi*0.08*100mm = 50.3 mm/s
ideal_speed = 2 * np.pi * 0.08 * 200
ax.axhline(ideal_speed, color='k', ls='--', alpha=0.3, label=f"Ideal {ideal_speed:.1f}")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (mm/s)")
ax.set_title("EE Speed (Smoothness)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6. Error distribution
ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(0, max(d_pd["err_mm"][skip:].max(),
                          d_rl["err_mm"][skip:].max()) + 0.3, 35)
ax.hist(d_pd["err_mm"][skip:], bins=bins, alpha=0.5, color='C0', label="PD", density=True)
ax.hist(d_rl["err_mm"][skip:], bins=bins, alpha=0.5, color='C3', label="PD+RL", density=True)
ax.axvline(3.0, color='gray', ls=':', alpha=0.5, label="3mm")
ax.set_xlabel("Error (mm)"); ax.set_ylabel("Density")
ax.set_title("Error Distribution")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 7. RL actions
ax = fig.add_subplot(gs[2, 1])
for i, lbl in enumerate(["X", "Y", "Z"]):
    ax.plot(d_rl["time"], d_rl["actions"][:,i] * 5, lw=0.8,
            label=f"RL {lbl} (mm)", alpha=0.7)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Correction (mm)")
ax.set_title("RL Actions (Task-Space Corrections)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 8. Metrics bar chart
ax = fig.add_subplot(gs[2, 2])
labels = ["Mean", "Max", "Std", "RMSE", "P95"]
pd_vals = [m_pd["mean_mm"], m_pd["max_mm"], m_pd["std_mm"],
           m_pd["rmse_mm"], m_pd["p95_mm"]]
rl_vals = [m_rl["mean_mm"], m_rl["max_mm"], m_rl["std_mm"],
           m_rl["rmse_mm"], m_rl["p95_mm"]]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, pd_vals, w, label="PD Only", color='C0', alpha=0.7)
ax.bar(x + w/2, rl_vals, w, label="PD+RL", color='C3', alpha=0.7)
ax.axhline(3.0, color='gray', ls=':', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Error (mm)"); ax.set_title("Metrics Comparison")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("step7_final_results.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print(f"\nPlot saved to step7_final_results.png")
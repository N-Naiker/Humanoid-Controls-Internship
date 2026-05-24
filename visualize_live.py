"""
Watch the trained RL agent control the Franka Panda in real-time.
"""
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from env import FrankaTrackingEnv

import os
import time

os.system('cls')

# ── Setup environment and load model ──
env = FrankaTrackingEnv(
    obs_noise_std=0.005,
    enable_delay=True,
    ctrl_substeps=10,
    residual_scale=0.005,
)
rl_model = PPO.load("models/ppo_best")

# ── Reset and get initial state ──
obs, _ = env.reset(seed=42)

# ── Launch viewer ──
# The red sphere is the trajectory target.
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 2
    viewer.cam.lookat[:] = [0.5, 0.0, 0.6]

    step = 0
    while viewer.is_running():
        # RL agent picks action
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        err = info["tracking_error_mm"]
        if step % 25 == 0:  
            print(f"t={env.sim_time:.1f}s  error={err:.2f}mm")

        # Sync viewer
        viewer.sync()

        step += 1
        if truncated:
            print(f"\nEpisode done. Restarting...")
            obs, _ = env.reset(seed=np.random.randint(1000))
            step = 0
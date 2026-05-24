"""
Render the trained RL agent to a video file.
Saves as MP4 for GitHub submission.
"""
import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from env import FrankaTrackingEnv
import os

os.system('cls')

# ── Setup ──
env = FrankaTrackingEnv(
    obs_noise_std=0.005,
    enable_delay=True,
    ctrl_substeps=10,
    residual_scale=0.005,
)
rl_model = PPO.load("models/ppo_best")

# Increase offscreen framebuffer for higher resolution rendering
env.model.vis.global_.offwidth = 1280
env.model.vis.global_.offheight = 720

# ── Create offscreen renderer ──
width, height = 1280, 720
renderer = mujoco.Renderer(env.model, height=height, width=width)

# Camera setup
camera = mujoco.MjvCamera()
camera.azimuth = 135
camera.elevation = -25
camera.distance = 2
camera.lookat[:] = [0.5, 0.0, 0.6]

# ── Run simulation and collect frames ──
obs, _ = env.reset(seed=42)
frames = []
render_every = 2 

print("Rendering video...")
for step in range(env.max_steps):
    action, _ = rl_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % render_every == 0:
        renderer.update_scene(env.data, camera)
        frame = renderer.render()
        frames.append(frame.copy())

    if step % 50 == 0:
        print(f"  t={env.sim_time:.1f}s  error={info['tracking_error_mm']:.2f}mm  "
              f"frames={len(frames)}")

    if truncated:
        break

# ── Save video ──
output_path = "tracking_demo.mp4"
fps = 25
writer = imageio.get_writer(output_path, fps=fps, quality=8)
for frame in frames:
    writer.append_data(frame)
writer.close()

print(f"\nVideo saved to {output_path}")
print(f"  Duration: {len(frames)/fps:.1f}s")
print(f"  Frames: {len(frames)}")
print(f"  Resolution: {width}x{height}")
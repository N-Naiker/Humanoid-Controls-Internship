"""
Train PPO agent for trajectory tracking.

Evaluate after each training batch, save the best.
"""
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import FrankaTrackingEnv

os.system('cls')

skip = 65
steps = 650
runtime = 400_000

def make_env(rank):
    def _init():
        env = FrankaTrackingEnv(
            obs_noise_std=0.005,
            enable_delay=True,
            ctrl_substeps=10,
            residual_scale=0.005,
        )
        return Monitor(env)
    return _init


def evaluate(env, model, n_steps=steps , seed=99):
    """Run one episode and return mean/max error."""
    obs, _ = env.reset(seed=seed)
    errors = []
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if i >= skip:  
            errors.append(info["tracking_error_mm"])
        if truncated:
            break
    errs = np.array(errors)
    return np.mean(errs), np.max(errs), np.std(errs)


def train(total_timesteps=100_000, n_envs=4, save_dir="models"):
    """Train PPO with periodic evaluation and early stopping."""

    # ── Create parallel training environments ──
    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # ── Create PPO agent ──
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,     
        n_steps=512,        
        batch_size=64,
        n_epochs=5,
        gamma=0.99,  
        gae_lambda=0.95,   
        clip_range=0.15,         
        ent_coef=0.001,   
        vf_coef=0.5,             
        max_grad_norm=0.5,      
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  
            activation_fn=torch.nn.Tanh,              
        ),
        verbose=0,
        seed=42,
    )

    # ── Evaluation environment ──
    eval_env = FrankaTrackingEnv(
        obs_noise_std=0.005,
        enable_delay=True,
        ctrl_substeps=10,
        residual_scale=0.005,
    )

    # ── PD-only baseline for comparison ──
    obs, _ = eval_env.reset(seed=99)
    pd_errors = []
    for i in range(eval_env.max_steps):
        obs, _, _, trunc, info = eval_env.step(np.zeros(3))
        if i >= skip:
            pd_errors.append(info["tracking_error_mm"])
        if trunc:
            break
    pd_mean = np.mean(pd_errors)
    pd_max  = np.max(pd_errors)

    print("=" * 60)
    print("  RL TRAINING")
    print("=" * 60)
    print(f"  PD-only baseline: mean={pd_mean:.3f}mm, max={pd_max:.3f}mm")
    print(f"  Total timesteps:  {total_timesteps}")
    print(f"  Parallel envs:    {n_envs}")
    print(f"  Eval every:       5,000 steps")
    print("-" * 60)
    print(f"  {'Step':>8}  {'Mean (mm)':>10}  {'Max (mm)':>10}  {'Std (mm)':>10}  {'Status':>8}")
    print("-" * 60)

    # ── Training loop with early stopping ──
    os.makedirs(save_dir, exist_ok=True)
    best_mean = float("inf")
    eval_interval = 5_000
    n_evals = total_timesteps // eval_interval

    for epoch in range(n_evals):
        # Train for one batch
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)

        # Evaluate
        mean_err, max_err, std_err = evaluate(eval_env, model)

        # Save best model
        status = ""
        if mean_err < best_mean:
            best_mean = mean_err
            model.save(os.path.join(save_dir, "ppo_best"))
            status = "← BEST"

        print(f"  {(epoch+1)*eval_interval:>8}  {mean_err:>10.3f}  {max_err:>10.3f}  "
              f"{std_err:>10.3f}  {status:>8}")

    train_env.close()

    print("-" * 60)
    print(f"  Best model: {best_mean:.3f}mm (saved to {save_dir}/ppo_best)")
    print(f"  PD baseline was: {pd_mean:.3f}mm")
    improvement = (pd_mean - best_mean) / pd_mean * 100
    print(f"  Improvement: {improvement:.1f}%")
    print("=" * 60)

    return model

if __name__ == "__main__":
    train(total_timesteps=runtime)
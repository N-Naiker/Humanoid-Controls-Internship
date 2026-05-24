# 3D End-Effector Trajectory Tracking — Franka Emika Panda
This system combines a Jacobian-based PD controller, Kalman filter, and PPO-trained residual policy to track a circular end-effector trajectory on a simulated Franka Panda in MuJoCo. The RL agent reduces mean tracking error by 67.8% compared to the classical controller alone, achieving 2.38mm accuracy under 5mm observation noise and 20ms control delay.

| Metric | PD Only | PD + RL | Improvement |
|--------|---------|---------|-------------|
| Mean error | 7.38 mm | 2.38 mm | -67.8% |
| Max error | 11.09 mm | 5.05 mm | -54.5% |

Video and Graphs Results can be found in the results folder

## Design Note
The design and execution of this task was done using root cause analysis asking what, why, and how.

What do I need to do? ---> Track robotic hand tracking circle in cartesian plane
What does this tell me I need?
      - Simulation Software ---> Mujoco as it is more lightweight for implementation and allows easy access to various robotic arm simulations
      - Target Path ---> Calculate using math class. For this case I generated a circle pathway in the yz-plane using sin and cos
      - Implement sources of Uncertainty ---> Chose two to make challenge more interesting, state noise and lag
      - How to filter noise from signal ---> Kalman Filter as can help estimate desired position and is robust (effectively a low pass filter)
      - Solve Tracking Error ---> Implement a PD controller for tracking position and orientation of end effector (No Integrator as cause oscillarions in sims)
      - Implement RL agent to get optimal performance and achieve smooth motion
      
### Architecture
```
Trajectory → [+ RL correction] → PD Controller → MuJoCo
                                       ↑
                                 Kalman Filter ← Noisy Sensor
```

The RL agent runs at 50Hz and outputs 3D task-space corrections that shift the PD controller's setpoint. The PD controller and Kalman filter run at 500Hz inside the simulation loop. Uncertainty sources: 5mm Gaussian observation noise (filtered by a tuned Kalman filter) and 20ms control delay. Orientation tracking uses the full 6×7 Jacobian.

### State, Action, and Reward
The **state** (14-dim) contains position error, velocity error, and orientation error from Kalman-filtered estimates, trajectory phase as (sin θ, cos θ), and the previous action. Observations are manually scaled so all components have similar magnitude.

The **action** (3-dim) is a task-space position correction scaled to +-5mm, applied as a residual on top of the PD controller. Task-space was chosen because the Jacobian pseudoinverse naturally smooths the output which was critical for eliminating jitter.

The **reward** combines a linear tracking penalty, a smoothness term penalising error changes, gentle action regularisation, an action rate penalty, and an orientation penalty. Linear tracking was chosen over exponential because the gradient stays useful at sub-millimetre errors. Action penalties are deliberately light as heavy penalties caused the agent to learn that inaction was optimal.

### Trajectory Representation
The trajectory was represented as a circle in the YZ-plane, with a radius of 200mm, a frequency of 0.08Hz, analytic position and velocity. Phase was encoded as sin θ and cos θ in the observation, giving the agent a continuous sense of position on the circle without discontinuities.

### Evaluating Tracking Performance
Euclidean position error (mean, max, RMSE, STD) measured against the target position at that time step. Smoothness is quantified by EE speed standard deviation and mean jerk. The first few seconds of each episode is excluded to avoid counting the settling transient. PD-only and PD+RL are compared under identical noise and delay conditions.

### Key Decisions
- **Residual RL** over end-to-end: the PD handles stability, RL refines accuracy. Converges in 400k steps instead of millions.
- **Softened actuator gains** (0.5× stiffness, 2× damping): reduces physical jitter from noise-driven corrections, at the cost of baseline precision that RL recovers.
- **Kalman filter tuned for in-loop use** (Q=1e-6): standard passive tuning caused controller instability due to filter lag.

### RL Training
- Algorithm: PPO (Stable-Baselines3)
- Network: 2 × 64 hidden layers, Tanh activation
- 400k timesteps, 4 parallel environments, early stopping

# Setup & Running Instructions
## Prerequisites
- Python 3.8+
- MuJoCo >= 3.0
- Franka Emika Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## Installation
```bash
git clone <your-repo-url>
cd humanoid-challenge
pip install -r requirements.txt
```

Place the `franka_emika_panda` folder from MuJoCo Menagerie inside `mujoco_menagerie/` so the path `mujoco_menagerie/franka_emika_panda/scene_modified.xml` is valid.
Additionally, when loding the model please ensure you use the scene_modified.xml file. It has been modified to use the pandas_nohand.xml file and places a red ball
to make it easier to track the target position in the video

Alternatively, just add the scene_modifier.xml file to the franka_emika_panda folder once you've installed mujoco_menagerie

## Running
```bash
# Train the RL agent from scratch
python train.py

# Evaluate pre-trained model and generate comparison plots
python evaluate.py

# Watch the robot track in real-time 
python visualize_live.py

# Record a demo video (Not needed for you but shows how I got the demo video)
python visualize_video.py
```

A pre-trained model is included in `models/ppo_best.zip`. Run `evaluate.py` or `visualize_live.py` directly without training first.

## Project Structure
```
├── env.py              # RL environment (Gymnasium)
├── train.py            # PPO training with early stopping
├── evaluate.py         # Evaluation + comparison plots
├── visualize_live.py   # Real-time MuJoCo viewer
├── visualize_video.py  # Record MP4 video
├── scene_modified.xml  # MuJoCo scene definition
├── requirements.txt
├── models/
│   └── ppo_best.zip    # Pre-trained model
└── results/
    └── tracking_results.png
```

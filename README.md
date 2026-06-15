# Feagine Gymnasium Environment

`FeagineReachEnv` wraps the Feagine MuJoCo demo API as a Gymnasium
environment for end-effector reaching tasks.

## Install

The Feagine controller requires both packages supplied in the Feagine release:

1. Build and install `feagine-simulation-core` with `ENABLE_PYTHON=ON`.
2. Install the supplied `feagine_mujoco-0.1.0-py3-none-any.whl`.
3. Install Gymnasium.

```powershell
python -m pip install path\to\feagine_mujoco-0.1.0-py3-none-any.whl
python -m pip install gymnasium
python -c "import pyfeagine_sim_core, feagine_mujoco; print('imports ok')"
```

The wheel installs its own `numpy>=1.24` and `mujoco` dependencies. MJCF is
recommended for training and control; URDF is mainly intended for loading and
inspection and has a simpler grasper representation.

## Action

The action is a normalized seven-dimensional `Box(-1, 1)`:

```text
[alpha_0, alpha_1, alpha_2, beta_0, beta_1, beta_2, gripper]
```

The first six values are scaled to the configured PCC angle limits. The final
value is mapped continuously from `[-1, 1]` to the Feagine grip command
`[0, 1]`, where `0` is open and `1` is closed.

## Observation

The observation is a `Dict` containing:

- `qpos`, `qvel`: MuJoCo generalized position and velocity
- `tip_position`, `tip_quaternion`: virtual tool-tip pose; quaternion order is
  `(w, x, y, z)`
- `target_position`: current reaching target
- `gripper`: applied Feagine grip command in `[0, 1]`

## Usage

```python
from feagine_env import FeagineEnvConfig, FeagineReachEnv

config = FeagineEnvConfig(
    model_type="mjcf",
    max_episode_steps=300,
    success_tolerance=0.015,
)
env = FeagineReachEnv(config, render_mode="human")
observation, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
```

Pass a fixed target to the constructor:

```python
env = FeagineReachEnv(target_position=[0.65, 0.0, 0.10])
```

Or set a target for one episode:

```python
observation, info = env.reset(
    options={"target_position": [0.65, 0.0, 0.10]}
)
```

Run the random-action viewer demo with:

```powershell
python demo_gym_env.py
```

## IK demo

`demo_ik_env.py` samples nearby, inward-biased targets and solves each one with
`FeagineRobotBase.solve_inverse_kinematics()`, converts the three solved PCC
sections to a normalized Gymnasium action, and steps the environment until the
tool tip reaches the target:

```powershell
python demo_ik_env.py
```

Use a fixed world-frame target or run without rendering:

```powershell
python demo_ik_env.py --target 0.65 0.0 0.10 --episodes 1
python demo_ik_env.py --episodes 20 --no-render
```

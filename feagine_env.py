from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - depends on user environment
    raise ModuleNotFoundError(
        "FeagineReachEnv requires Gymnasium. Install it with `pip install gymnasium`."
    ) from exc

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - depends on user environment
    raise ModuleNotFoundError(
        "FeagineReachEnv requires MuJoCo. Install it with `pip install mujoco`."
    ) from exc

try:
    from feagine_mujoco import (
        FeagineDeformableVisual,
        FeagineMjcfRobot,
        FeagineUrdfRobot,
        asset_path,
        demo_scene_path,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on user environment
    raise ModuleNotFoundError(
        "FeagineReachEnv requires the `feagine_mujoco` package."
    ) from exc


@dataclass(frozen=True)
class FeagineEnvConfig:
    """Configuration for :class:`FeagineReachEnv`."""

    model_type: str = "mjcf"
    frame_skip: int = 4
    max_episode_steps: int = 300
    success_tolerance: float = 0.015
    target_radius: float = 0.06
    alpha_limit: float = 0.65
    beta_limit: float = 0.45
    action_penalty: float = 0.01

    def __post_init__(self) -> None:
        if self.model_type not in {"mjcf", "urdf"}:
            raise ValueError("model_type must be either 'mjcf' or 'urdf'")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        if self.success_tolerance <= 0.0:
            raise ValueError("success_tolerance must be positive")
        if self.target_radius < 0.0:
            raise ValueError("target_radius cannot be negative")


class FeagineReachEnv(gym.Env):
    """Gymnasium environment for PCC control of the Feagine robot.

    Action layout:
        ``[alpha_0, alpha_1, alpha_2, beta_0, beta_1, beta_2, gripper]``

    All actions are normalized to ``[-1, 1]``. The six arm values are scaled
    by the configured angle limits. The gripper value is linearly mapped from
    ``[-1, 1]`` to the package's continuous grip command range ``[0, 1]``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    _TARGET_MARKER_RADIUS = 0.008
    _TARGET_MARKER_COLOR = np.asarray(
        [1.0, 0.2, 0.0, 1.0],
        dtype=np.float32,
    )
    _TARGET_HALO_COLOR = np.asarray(
        [1.0, 0.85, 0.0, 0.25],
        dtype=np.float32,
    )

    def __init__(
        self,
        config: FeagineEnvConfig | None = None,
        *,
        render_mode: str | None = None,
        target_position: np.ndarray | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()
        self.config = config or FeagineEnvConfig()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)

        model_path = demo_scene_path(
            model_type=self.config.model_type,
            scene_variant="single",
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Feagine scene asset: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        robot_class = (
            FeagineMjcfRobot
            if self.config.model_type == "mjcf"
            else FeagineUrdfRobot
        )
        self.robot = robot_class(self.model, self.data)
        self._deformable_visual: FeagineDeformableVisual | None = None
        if self.config.model_type == "urdf":
            self._deformable_visual = FeagineDeformableVisual(
                model=self.model,
                data=self.data,
                mesh_path=asset_path("meshes", "feagine_arm_mesh.obj"),
                base_body_name="base_link",
                segment_body_prefix="capsule_",
                mesh_name="feagine_arm_mesh",
            )
            self.model.geom_contype[self._deformable_visual.mesh_geom_id] = 0
            self.model.geom_conaffinity[self._deformable_visual.mesh_geom_id] = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.model.nq,),
                    dtype=np.float64,
                ),
                "qvel": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.model.nv,),
                    dtype=np.float64,
                ),
                "tip_position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float64,
                ),
                "tip_quaternion": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float64,
                ),
                "target_position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float64,
                ),
                "gripper": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self._fixed_target = (
            None
            if target_position is None
            else np.asarray(target_position, dtype=np.float64).reshape(3).copy()
        )
        self.target_position = np.zeros(3, dtype=np.float64)
        self._gripper_command = 0.0
        self._elapsed_steps = 0
        self._viewer: Any | None = None
        self._renderer: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.robot.set_pcc_angles(alphas=[0.0] * 3, betas=[0.0] * 3)
        self.robot.open()
        mujoco.mj_forward(self.model, self.data)
        self._update_deformable_visual()
        self._elapsed_steps = 0
        self._gripper_command = 0.0

        options = options or {}
        option_target = options.get("target_position")
        if option_target is not None:
            self.target_position = np.asarray(
                option_target, dtype=np.float64
            ).reshape(3).copy()
        elif self._fixed_target is not None:
            self.target_position = self._fixed_target.copy()
        else:
            tip_position = self._tip_pose()[0]
            self.target_position = tip_position + self.np_random.uniform(
                low=-self.config.target_radius,
                high=self.config.target_radius,
                size=3,
            )

        observation = self._get_observation()
        info = self._get_info(observation["tip_position"])
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        alphas = action[:3] * self.config.alpha_limit
        betas = action[3:6] * self.config.beta_limit
        self.robot.set_pcc_angles(alphas=alphas, betas=betas)
        self._gripper_command = 0.5 * (float(action[6]) + 1.0)
        self.robot.set_grip_command(self._gripper_command)

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._update_deformable_visual()
        self._elapsed_steps += 1

        observation = self._get_observation()
        info = self._get_info(observation["tip_position"])
        terminated = bool(info["is_success"])
        truncated = self._elapsed_steps >= self.config.max_episode_steps
        reward = -float(info["distance"]) - self.config.action_penalty * float(
            np.square(action[:6]).mean()
        )
        if terminated:
            reward += 1.0

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
                self._viewer.cam.lookat[:] = [0.68, 0.0, 0.08]
                self._viewer.cam.distance = 0.25
                self._viewer.cam.azimuth = 160.0
                self._viewer.cam.elevation = -16.0
            self._update_deformable_visual(self._viewer)
            self._update_target_marker()
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self.height,
                    width=self.width,
                )
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.robot.tip_pose()
        return (
            np.asarray(pose.position, dtype=np.float64).copy(),
            np.asarray(pose.quaternion, dtype=np.float64).copy(),
        )

    def _update_deformable_visual(self, viewer: Any | None = None) -> None:
        if self._deformable_visual is not None:
            self._deformable_visual.update(viewer)

    def _update_target_marker(self) -> None:
        if self._viewer is None:
            return

        identity_rotation = np.eye(3, dtype=np.float64).reshape(-1)
        with self._viewer.lock():
            scene = self._viewer.user_scn
            if len(scene.geoms) < 2:
                raise RuntimeError(
                    "MuJoCo viewer user scene does not have room for target markers."
                )

            mujoco.mjv_initGeom(
                scene.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[
                    2.0 * self._TARGET_MARKER_RADIUS,
                    0.0,
                    0.0,
                ],
                pos=self.target_position,
                mat=identity_rotation,
                rgba=self._TARGET_HALO_COLOR,
            )
            mujoco.mjv_initGeom(
                scene.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[
                    self._TARGET_MARKER_RADIUS,
                    0.0,
                    0.0,
                ],
                pos=self.target_position,
                mat=identity_rotation,
                rgba=self._TARGET_MARKER_COLOR,
            )
            scene.ngeom = 2

    def _get_observation(self) -> dict[str, np.ndarray]:
        tip_position, tip_quaternion = self._tip_pose()
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "tip_position": tip_position,
            "tip_quaternion": tip_quaternion,
            "target_position": self.target_position.copy(),
            "gripper": np.asarray(
                [self._gripper_command],
                dtype=np.float32,
            ),
        }

    def _get_info(self, tip_position: np.ndarray) -> dict[str, Any]:
        distance = float(np.linalg.norm(tip_position - self.target_position))
        return {
            "distance": distance,
            "is_success": distance <= self.config.success_tolerance,
            "elapsed_steps": self._elapsed_steps,
        }

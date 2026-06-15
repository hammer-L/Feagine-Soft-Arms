from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np

from feagine_env import FeagineEnvConfig, FeagineReachEnv
from feagine_mujoco import virtual_ee_axis_from_quaternion_wxyz


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Feagine inverse kinematics to drive the Gymnasium environment."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of targets to solve. Defaults to 5.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Use one fixed world-frame target instead of sampled targets.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    return parser


def ik_sections_to_action(
    sections: Sequence[object],
    config: FeagineEnvConfig,
    *,
    grip_command: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Convert an IK solution into the environment's normalized action."""
    if len(sections) != 3:
        raise ValueError(f"Expected 3 IK sections, got {len(sections)}")
    if not 0.0 <= grip_command <= 1.0:
        raise ValueError("grip_command must be in [0, 1]")

    raw_action = np.asarray(
        [
            *(float(section.alpha) / config.alpha_limit for section in sections),
            *(float(section.beta) / config.beta_limit for section in sections),
            2.0 * grip_command - 1.0,
        ],
        dtype=np.float32,
    )
    was_clipped = bool(np.any(np.abs(raw_action) > 1.0))
    return np.clip(raw_action, -1.0, 1.0), was_clipped


def sample_nearby_target(
    tip_position: np.ndarray,
    tip_direction: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an inward-biased nearby target to avoid overextending the arm."""
    direction = np.asarray(tip_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)

    lateral = rng.normal(size=3)
    lateral -= np.dot(lateral, direction) * direction
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1.0e-9:
        fallback = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, direction))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0])
        lateral = np.cross(direction, fallback)
        lateral_norm = float(np.linalg.norm(lateral))
    lateral /= lateral_norm

    inward_distance = rng.uniform(0.015, 0.035)
    lateral_distance = rng.uniform(0.005, 0.025)
    return (
        np.asarray(tip_position, dtype=np.float64)
        - inward_distance * direction
        + lateral_distance * lateral
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    config = FeagineEnvConfig(
        model_type="mjcf",
        target_radius=0.04,
        success_tolerance=0.01,
        max_episode_steps=240,
    )
    env = FeagineReachEnv(
        config,
        render_mode=None if args.no_render else "human",
        target_position=args.target,
    )

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=episode)
            tip_direction = virtual_ee_axis_from_quaternion_wxyz(
                observation["tip_quaternion"]
            )
            if args.target is None:
                target = sample_nearby_target(
                    observation["tip_position"],
                    tip_direction,
                    np.random.default_rng(episode),
                )
                observation, _ = env.reset(
                    seed=episode,
                    options={"target_position": target},
                )
            else:
                target = observation["target_position"]

            ik_result = env.robot.solve_inverse_kinematics(
                target_position=target,
                target_tip_direction=tip_direction,
                apply_solution=False,
                use_virtual_ee_offset=True,
            )
            action, was_clipped = ik_sections_to_action(
                ik_result.solved_sections,
                config,
                grip_command=0.0,
            )

            print(
                f"[episode {episode}] target={np.round(target, 5)} "
                f"ik_success={ik_result.success} "
                f"iterations={ik_result.iterations} "
                f"ik_position_error={ik_result.final_position_error:.6f}"
            )
            if was_clipped:
                print(
                    "  Warning: the IK solution exceeded the environment's "
                    "PCC angle limits and was clipped."
                )

            while True:
                observation, reward, terminated, truncated, info = env.step(action)
                if not args.no_render:
                    time.sleep(1.0 / env.metadata["render_fps"])
                if terminated or truncated:
                    print(
                        f"  finished: success={info['is_success']} "
                        f"distance={info['distance']:.6f} "
                        f"steps={info['elapsed_steps']} reward={reward:.4f}"
                    )
                    break

            if not args.no_render:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from feagine_env import FeagineReachEnv


def main() -> None:
    env = FeagineReachEnv(render_mode="human")
    observation, info = env.reset(seed=42)
    print("Initial tip:", observation["tip_position"])
    print("Target:", observation["target_position"])

    try:
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(
                    f"Episode ended: success={info['is_success']} "
                    f"distance={info['distance']:.4f} reward={reward:.4f}"
                )
                observation, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()

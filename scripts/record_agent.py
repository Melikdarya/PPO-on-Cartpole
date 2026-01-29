import gymnasium as gym
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pathlib import Path
from src.ppo import PPOAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_episodes(agent: PPOAgent, num_episodes=1, folder_path="recordings", name_prefix="eval-"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=folder_path,  # Folder to save videos
        name_prefix=name_prefix,            # prefix for video filenames
        episode_trigger=lambda x: True  # record every episode
    )

    folder_path = Path(folder_path)
    folder_path.mkdir(exist_ok=True, parents=True)

    for i in range(num_episodes):
        state, info = env.reset()
        episode_over = False

        while not episode_over:
            state = torch.tensor(state, dtype=torch.float32, device=device)
            action = agent.get_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

    env.close()


def record_random_behavior(num_episodes=1, folder_path="recordings", name_prefix="random-"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=folder_path,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True
    )

    folder_path = Path(folder_path)
    folder_path.mkdir(exist_ok=True, parents=True)

    for i in range(num_episodes):
        state, info = env.reset()
        episode_over = False

        while not episode_over:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

    env.close()


if __name__ == '__main__':
    # Agent = PPOAgent(4, 2, device)
    # Agent.load_model_from_dict("../models/Agent_0.pth")
    # record_episodes(Agent, 3, "../recordings", name_prefix="Agent_0")
    record_random_behavior(5, "../recordings", "random")

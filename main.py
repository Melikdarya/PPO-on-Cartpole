import torch
import sys
from src.env import CartPoleEnv
from src.ppo import PPOAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------> load <----------

test_env = CartPoleEnv(device, "human")
newAgent = PPOAgent(test_env.observation_space, test_env.action_space, device)
newAgent.load_model_from_dict("models/Agent_0.pth")


# ----------> test <----------

test_episodes = 5
rewards = newAgent.test(test_env, test_episodes)
print(rewards)

import torch
from src.env import CartPoleEnv
from src.ppo import PPOAgent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
learning_rate = 0.001
discount_factor = 0.99
# TODO
...

training_episodes = 100
display_episodes = 5

# training
train_env = CartPoleEnv(device)
agent = PPOAgent(train_env, learning_rate, 10, 10, 100, discount_factor,
                 0.9, 0.1, device)
# TODO: training CartPole agent

# testing
test_env = CartPoleEnv(device, "human")
rewards = agent.test(test_env, 5)
print(rewards)

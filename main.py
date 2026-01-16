import torch
from src.env import CartPoleEnv
from src.ppo import PPOAgent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
actor_learning_rate = 0.001
critic_learning_rate = 0.001
num_epochs = 20
minibatch_size = 64
steps_per_rollout = 100
discount_factor = 0.99  # gamma
gae_lambda = 0.95
entropy_bonus_coef = 0.01
epsilon = 0.2  # clipping


# ----------> train <----------

train_env = CartPoleEnv(device)
CartPoleAgent = PPOAgent(train_env.observation_space, train_env.action_space, device)
CartPoleAgent.train(train_env, actor_learning_rate, critic_learning_rate, num_epochs, minibatch_size, steps_per_rollout,
                    entropy_bonus_coef, discount_factor, gae_lambda, epsilon)


# ----------> save <----------

CartPoleAgent.save_model_parameters("myCartPoleAgent.pth")


# ----------> load <----------

newAgent = PPOAgent(train_env.observation_space, train_env.action_space, device)
newAgent.load_model_from_dict("myCartPoleAgent.pth")


# ----------> test <----------

display_episodes = 5
test_env = CartPoleEnv(device, "human")
rewards = newAgent.test(test_env, 5)
print(rewards)

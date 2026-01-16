import torch
from src.env import CartPoleEnv
from src.ppo import PPOAgent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
actor_learning_rate = 0.001
critic_learning_rate = 0.001
num_epochs = 20
minibatch_size = 64
steps_per_epoch = 100
discount_factor = 0.99  # gamma
gae_lambda = 0.95
entropy_bonus_coef = 0.01
epsilon = 0.2  # clipping

# control bits

# ----------> train <----------

train_env = CartPoleEnv(device)
agent = PPOAgent(train_env, actor_learning_rate, critic_learning_rate, num_epochs, minibatch_size,
                 steps_per_epoch, entropy_bonus_coef, discount_factor, gae_lambda, epsilon, device)
agent.train()


# ----------> save <----------

agent.save_model_parameters("test_dummy_model")


# ----------> load <----------

agent.load_model_from_dict("test_dummy_model")


# ----------> test <----------

display_episodes = 5
test_env = CartPoleEnv(device, "human")
rewards = agent.test(test_env, 5)
print(rewards)

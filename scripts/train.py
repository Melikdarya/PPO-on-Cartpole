import torch
from src.env import CartPoleEnv
from src.ppo import PPOAgent


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparameters
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    num_epochs = 100
    minibatch_size = 64
    steps_per_rollout = 250
    discount_factor = 0.99  # gamma
    gae_lambda = 0.95
    entropy_bonus_coef = 0.01
    epsilon = 0.2  # clipping

    train_env = CartPoleEnv(device)

    CartPoleAgent = PPOAgent(train_env.observation_space, train_env.action_space, device)
    CartPoleAgent.train(train_env, actor_learning_rate, critic_learning_rate, num_epochs, minibatch_size,
                        steps_per_rollout, entropy_bonus_coef, discount_factor, gae_lambda, epsilon)

    CartPoleAgent.save_model_parameters("../models/Agent_0.pth")


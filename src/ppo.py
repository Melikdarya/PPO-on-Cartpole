import numpy as np
import torch
from src.env import CartPoleEnv
from src.models import Actor, Critic
from src.rollout import RolloutBuffer, collect_rollout


class PPOAgent:
    def __init__(self,
                 env: CartPoleEnv,
                 learning_rate: float,
                 batch_size: int,
                 epochs: int,
                 steps_per_rollout: int,
                 gamma: float,
                 gae_lambda: float,
                 epsilon: float,
                 device: torch.device):
        self.env = env
        self.learning_rate = learning_rate
        self.steps_per_rollout = steps_per_rollout
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.device = device

        self.actor = Actor(env.observation_space, env.action_space).to(device)
        self.critic = Critic(env.observation_space).to(device)
        self.buffer = RolloutBuffer()

    def _calc_surrogate_loss(self):
        pass
        # TODO

    def train(self) -> None:
        pass
        # TODO
        # for epoch in range(self.epochs):

    def test(self, test_env: CartPoleEnv, display_episodes: int) -> list:
        assert test_env.observation_space == self.env.observation_space
        assert test_env.action_space == self.env.action_space

        self.actor.eval()

        total_rewards = []
        for i in range(display_episodes):
            state = test_env.reset()
            episode_over = False
            total_reward = 0

            while not (episode_over):
                with torch.inference_mode():
                    action_logits = self.actor(state)

                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1)
                state, reward, episode_over, _ = test_env.step(action.item())
                total_reward += reward

            total_rewards.append(total_reward)

        return total_rewards


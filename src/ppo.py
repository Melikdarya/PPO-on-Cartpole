import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from pathlib import Path

from src.env import CartPoleEnv
from src.models import Actor, Critic
from src.rollout import RolloutBuffer, collect_rollout


class PPOAgent:
    """
    TODO: class description
    """
    def __init__(self,
                 env_observaiotn_space_size: int,
                 env_action_space_size: int,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        self.observation_space_size = env_observaiotn_space_size
        self.action_space_size = env_action_space_size

        self.actor = Actor(env_observaiotn_space_size, env_action_space_size).to(device)
        self.device = device

    def _calc_surrogate_loss(self, action_logits_new: torch.Tensor,
                             action_logits_old: torch.Tensor,
                             advantages: torch.Tensor,
                             epsilon: float) -> torch.Tensor:
        """
        Calculate surrogate loss using objective: E[min(r(θ)*A, clip(r(θ)*A, 1-ϵ, 1+ϵ))].
        """
        advantages = advantages.detach()  # treat as constants, do not compute gradients
        prob_ration = (action_logits_new - action_logits_old).exp()
        loss = torch.min(
            prob_ration * advantages,
            torch.clip(prob_ration, 1.0 - epsilon, 1.0 + epsilon)  # torch.clip() is alias for torch.clamp()
        )
        return -torch.mean(loss)  # empirical average, negative for gradient ascent

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Calculate new logprobs for collected states and taken actions, and policy entropy.
        """
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)   # get only logprobs of taken actions
        entropy = dist.entropy()            # how "deterministic" the model was at each step
        return logprobs, entropy

    def train(self,
              env: CartPoleEnv,
              actor_learning_rate: float,
              critic_learning_rate: float,
              epochs: int,
              batch_size: int,
              steps_per_rollout: int,
              entropy_bonus_coef: float = 0.01,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              epsilon: float = 0.2) -> None:
        """
        Proximal Policy Optimization algorithm with the following components:
        - actor-critic approach
        - Generalized Advantage Estimation (GAE)
        - clipped surrogate loss with clipping parameter ϵ
        - Mini-batch updates
        - entropy bonus
        """

        buffer = RolloutBuffer(device=self.device)

        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        critic = Critic(self.observation_space_size).to(self.device)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)
        critic_loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):

            # collect rollout data
            collect_rollout(env, self.actor, critic, buffer, steps_per_rollout, gamma, gae_lambda)

            self.actor.train()
            critic.train()

            # minibatch update
            for states, actions, logprobs, returns, advantages in buffer.get_batches(batch_size):
                # forward pass, calculate new logprobs
                new_logprobs, entropy = self._evaluate_actions(states, actions)
                values = critic(states)

                # calculate loss
                actor_loss = self._calc_surrogate_loss(new_logprobs, logprobs, advantages, epsilon)
                entropy_bonus = entropy_bonus_coef * entropy.mean()
                actor_loss = actor_loss - entropy_bonus

                critic_loss = critic_loss_fn(returns, values)

                # Optimizer zero grad
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

                # Backpropagate loss
                actor_loss.backward()
                critic_loss.backward()

                # Optimizer step
                actor_optimizer.step()
                critic_optimizer.step()

            # TODO: Testing loop goes here

    def test(self, test_env: CartPoleEnv, display_episodes: int) -> list:
        """
        TODO: description
        """

        self.actor.eval()

        total_rewards = []
        for i in range(display_episodes):
            state = test_env.reset()
            episode_over = False
            total_reward = 0

            while not episode_over:
                with torch.inference_mode():
                    action_logits = self.actor(state)

                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1)
                state, reward, episode_over, _ = test_env.step(action.item())
                total_reward += reward

            total_rewards.append(total_reward)

        return total_rewards

    def save_model_parameters(self, model_name: str) -> None:
        """
        Save policy network's state dict (learned parameters) under given model_name.

        Warning: If there exists another file at the same path, it will be overwritten.

        :param model_name: Model parameters will be saved to the model/model_name path.
        """
        folder_path = Path("models")
        model_name = Path(model_name)

        model_save_path = folder_path / model_name
        print(f"Saving model to: {model_save_path}")
        torch.save(self.actor.state_dict(), model_save_path)

    def load_model_from_dict(self, model_name: str) -> None:
        """
        Load's policy network's parameters from a saved model dictionary.

        :param model_name: Policy network's parameters will be loaded from path "models/model_name".
        """
        model_load_path = Path("models") / Path(model_name)
        self.actor.load_state_dict(torch.load(f=model_load_path, weights_only=True))
        print("All keys matched successfully!")


import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from pathlib import Path
import csv
import json
import os
from datetime import datetime

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

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            buffer.clear()

    def get_action(self, state: torch.Tensor):
        with torch.inference_mode():
            action_logits = self.actor(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        return action.item()

    def test(self,
             test_env: CartPoleEnv,
             test_episodes: int,
             csv_path: str | None = None,
             success_threshold: float | None = None,
             model_name: str | None = None) -> dict:
        """
        Run evaluation episodes and optionally save results to CSV.

        Returns a dict with keys: `rewards` (list), `mean`, `std`, `success_rate` (or None).
        If `csv_path` is provided the results will be appended to that CSV file. The CSV
        will include a header if the file does not exist.
        """

        self.actor.eval()

        total_rewards = []
        for i in range(test_episodes):
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

        # compute aggregate metrics
        rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32)
        mean = float(rewards_tensor.mean().item())
        std = float(rewards_tensor.std().item())
        success_rate = None
        if success_threshold is not None:
            success_rate = float((rewards_tensor >= float(success_threshold)).sum().item() / len(rewards_tensor))

        results = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name or "",
            "device": str(self.device),
            "episodes": int(test_episodes),
            "mean": mean,
            "std": std,
            "success_threshold": success_threshold if success_threshold is not None else "",
            "success_rate": success_rate if success_rate is not None else "",
            "rewards": total_rewards,
        }

        # Optionally write to CSV
        if csv_path is not None:
            # ensure directory exists
            dir_name = os.path.dirname(csv_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "model", "device", "episodes", "mean", "std", "success_threshold", "success_rate", "rewards"])
                if write_header:
                    writer.writeheader()
                # serialize rewards as JSON string
                row = results.copy()
                row["rewards"] = json.dumps(row["rewards"])
                writer.writerow(row)

        return results

    def save_model_parameters(self, path: str) -> None:
        """
        Save policy network's state dict (learned parameters) under given model_name.

        Warning: If there exists another file at the same path, it will be overwritten.

        :param path: Model parameters will be saved to this path.
        """
        model_save_path = Path(path)

        print(f"Saving model to: {model_save_path}")
        torch.save(self.actor.state_dict(), model_save_path)

    def load_model_from_dict(self, path: str) -> None:
        """
        Load's policy network's parameters from a saved model dictionary.

        :param path: Policy network's parameters will be loaded from this path.
        """
        model_load_path = Path(path)
        self.actor.load_state_dict(torch.load(f=model_load_path, weights_only=True))
        print("All keys matched successfully!")


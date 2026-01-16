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
    TODO: description
    """
    def __init__(self,
                 env: CartPoleEnv,
                 actor_learning_rate: float,
                 critic_learning_rate: float,
                 epochs: int,
                 batch_size: int,
                 steps_per_rollout: int,
                 entropy_bonus_coef: float = 0.01,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 epsilon: float = 0.2,
                 device: torch.device = torch.device('cpu')):
        # setup
        self.env = env
        self.device = device

        # tunable hyperparameters
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_rollout = steps_per_rollout
        self.entropy_bonus_coef = entropy_bonus_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon

        # actor network
        self.actor = Actor(env.observation_space, env.action_space).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        # critic network
        self.critic = Critic(env.observation_space).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.critic_loss_fn = nn.MSELoss()

        # rollout buffer
        self.buffer = RolloutBuffer(device=device)

    def _calc_surrogate_loss(self, action_logits_new: torch.Tensor,
                             action_logits_old: torch.Tensor,
                             advantages: torch.Tensor) -> torch.Tensor:
        """
        TODO: description
        """
        advantages = advantages.detach()  # treat as constants, do not compute gradients
        prob_ration = (action_logits_new - action_logits_old).exp()
        loss = torch.min(
            prob_ration * advantages,
            torch.clip(prob_ration, 1.0 - self.epsilon, 1.0 + self.epsilon)  # torch.clip() is alias for torch.clamp()
        )
        return -torch.mean(loss)  # empirical average, negative for gradient ascent

    def _evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        TODO: description
        """
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)   # get only logprobs of taken actions
        entropy = dist.entropy()            # how "deterministic" the model was at each step
        return logprobs, entropy

    def train(self) -> None:
        """
        TODO: description
        """

        for epoch in tqdm(range(self.epochs), desc="Training Progress", unit="epoch"):

            # start training loop
            # collect rollout data
            collect_rollout(self.env, self.actor, self.critic, self.buffer, self.steps_per_rollout)

            self.actor.train()
            self.critic.train()

            # minibatch update
            for states, actions, logprobs, returns, advantages in self.buffer.get_batches(self.batch_size):
                # calculate new logprobs (in first iteration should be the same)
                # forward pass
                new_logprobs, entropy = self._evaluate_actions(states, actions)
                values = self.critic(states)

                # calculate loss
                actor_loss = self._calc_surrogate_loss(new_logprobs, logprobs, advantages)
                entropy_bonus = self.entropy_bonus_coef * entropy.mean()
                actor_loss = actor_loss - entropy_bonus

                critic_loss = self.critic_loss_fn(returns, values)

                # Optimizer zero grad
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Backpropagate loss
                actor_loss.backward()
                critic_loss.backward()

                # Optimizer step
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            # TODO: Testing loop goes here

    def test(self, test_env: CartPoleEnv, display_episodes: int) -> list:
        """
        TODO: description
        """
        assert test_env.observation_space == self.env.observation_space  # TODO: should I check it?
        assert test_env.action_space == self.env.action_space

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

        Note: Function saves only actor's parameters. Critic's wages are not saved as they are not
        needed for making predictions.

        Warning: If there exists another file at the same path, it will be overwritten.

        :param model_name: Model parameters will be saved to the model/model_name.pth path.
        """
        folder_path = Path("models")
        model_name = Path(model_name)

        model_save_path = folder_path / model_name
        print(f"Saving model to: {model_save_path}")
        torch.save(self.actor.state_dict(), model_save_path)

    def load_model_from_dict(self, model_name: str) -> None:
        """
        Load's policy network's parameters from a saved model dictionary.

        :param model_name: Policy network's parameters will be loaded from path "models/model_name.pth".
        """
        model_load_path = Path("models") / Path(model_name)
        self.actor.load_state_dict(torch.load(f=model_load_path, weights_only=True))
        print("All keys matched successfully!")


import gymnasium as gym
import torch


class CartPoleEnv:
    def __init__(self, device='cpu', render_mode=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.device = device

    def reset(self):
        """
        Resets the environment to start a new episode.
        Returns the initial state as a PyTorch tensor.
        """
        state, info = self.env.reset()
        return self._to_tensor(state)

    def step(self, action):
        """
        Takes a step in the environment.
        Input: action (integer 0 or 1)
        Returns: next_state, reward, done, info
        """
        # If the action is a tensor, extract the integer value
        if isinstance(action, torch.Tensor):
            action = action.item()

        next_state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        return self._to_tensor(next_state), reward, done, info

    def _to_tensor(self, state):
        """Helper to convert numpy array to PyTorch tensor"""
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space.shape[0]

    @property
    def action_space(self):
        return self.env.action_space.n

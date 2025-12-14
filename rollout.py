
import torch

"""
Rollout collection for PPO.

This module is responsible for:
- Interacting with the CartPole environment
- Collecting trajectories using the current policy
- Storing transitions in RolloutBuffer
"""

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = None
        self.advantages = None

    def clear(self):
        """Clears the buffer after an update step"""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]

    def add(self, state, action, reward, done, logprob, value):
        """
        Store a transition.
        To be called after every step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def get(self):
        """
        Return all stored data as tensors.
        Ensures correct tensor shapes for the PPO update.
        """

        # .detach() removes the history of how we got here
        # Stack aligns all single tensors into one block
        batch_states = torch.stack(self.states).detach()
        batch_actions = torch.stack(self.actions).detach()
        batch_logprobs = torch.stack(self.logprobs).detach()
        batch_rewards = torch.tensor(self.rewards).detach()
        batch_dones = torch.tensor(self.dones, dtype=torch.float32).detach()
        batch_values = torch.stack(self.values).detach()

        return batch_states, batch_actions, batch_logprobs, batch_rewards, batch_dones, batch_values

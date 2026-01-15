import numpy as np
import torch
from torch.distributions import Categorical

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
        batch_actions = torch.tensor(self.actions, dtype=torch.long).detach()
        batch_logprobs = torch.tensor(self.logprobs, dtype=torch.float32).detach()
        batch_rewards = torch.tensor(self.rewards, dtype=torch.float32).detach()
        batch_dones = torch.tensor(self.dones, dtype=torch.float32).detach()
        batch_values = torch.tensor(self.values, dtype=torch.float32).detach()

        return (
            batch_states,
            batch_actions,
            batch_logprobs,
            batch_rewards,
            batch_dones,
            batch_values,
        )

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.
        GAE estimates how much better an action was compared to what the critic expected,
        while smoothly propagating future credit backward in time.

        next_value: float
            Value estimate V(s_{T+1}) for the state after the last rollout step.
        """
        # Convert to tensors
        values = torch.tensor(self.values + [next_value], dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros(len(rewards), dtype=torch.float32)

        gae = 0.0

        # Backward computation
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]  # mask dones to not take into account future value estimates from new sessions

            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae

            advantages[t] = gae

        # Store advantages and returns
        self.advantages = advantages
        self.returns = advantages + values[:-1]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1e-8
        )

    def get_batches(self, batch_size):
        """
        Yields small batches of data for training.
        """
        if self.returns is None:
            raise ValueError("You must call compute_gae() before getting batches!")

        # Gather all data into big tensors
        # We detach() to stop gradients from flowing back into the data collection phase
        states_tensor = torch.stack(self.states).detach()
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        logprobs_tensor = torch.tensor(self.logprobs, dtype=torch.float32)
        returns_tensor = self.returns.detach()
        advantages_tensor = self.advantages.detach()

        batch_count = len(self.states)
        indices = np.arange(batch_count)
        np.random.shuffle(indices) # Shuffle data to break correlations

        for start in range(0, batch_count, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                states_tensor[batch_idx],
                actions_tensor[batch_idx],
                logprobs_tensor[batch_idx],
                returns_tensor[batch_idx],
                advantages_tensor[batch_idx]
            )


def collect_rollout(env, actor, critic, buffer, steps_per_rollout):
    """
    Collects a batch of data using separate Actor and Critic networks.
    """
    state = env.reset()

    # Switch both to evaluation mode
    actor.eval()
    critic.eval()

    for step in range(steps_per_rollout):

        with torch.no_grad():
            # Prepare State (Batch of 1)
            t_state = state.unsqueeze(0)

            # Actor: Decides Action
            dist = Categorical(logits=actor(t_state))
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Critic: Estimates Value
            value = critic(t_state)

        # Step Environment
        # Action is a tensor [1], we need the int item
        next_state, reward, done, info = env.step(action.item())

        # Store Data
        buffer.add(
            state=state,
            action=action.item(),
            reward=reward,
            done=done,
            logprob=log_prob.item(),
            value=value.item()
        )

        state = next_state
        if done:
            state = env.reset()

    # Bootstrap Value for GAE
    with torch.no_grad():
        next_value = critic(state.unsqueeze(0))

    buffer.compute_gae(next_value.item())

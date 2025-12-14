import torch
from env import CartPoleEnv
from rollout import RolloutBuffer

def collect_rollout(env, buffer, steps=200):
    """
    Simulates the loop of collecting data.
    """
    state = env.reset()

    for _ in range(steps):
        # Random action: 0 (left) or 1 (right)
        action = env.env.action_space.sample()

        # Fake LogProb (e.g., 50% chance = log(0.5) = -0.69)
        log_prob = -0.693

        # Fake Value Estimate (Critic guesses the score is 10.0)
        value_estimate = 10.0

        # 2. Step the Environment
        next_state, reward, done, info = env.step(action)

        # 3. Store Data
        buffer.add(state, action, reward, done, log_prob, value_estimate)

        state = next_state

        if done:
            state = env.reset()

    print("Rollout complete!")


if __name__ == "__main__":
    # Initialize
    my_env = CartPoleEnv(render_mode="human")
    my_buffer = RolloutBuffer()

    # Run loop
    collect_rollout(my_env, my_buffer, steps=50)

    # Check shapes
    states, actions, logprobs, rewards, dones, values = my_buffer.get()

    print("\n--- Tensor Shape Check ---")
    print(f"States:   {states.shape}  (Expected: [50, 4])")
    print(f"Actions:  {actions.shape}  (Expected: [50])")
    print(f"Rewards:  {rewards.shape}  (Expected: [50])")
    print(f"Values:   {values.shape}   (Expected: [50])")

    if states.shape[1] == 4 and rewards.shape[0] == 50:
        print("\n SUCCESS: Infrastructure is ready.")
    else:
        print("\n ERROR: Shapes are incorrect.")
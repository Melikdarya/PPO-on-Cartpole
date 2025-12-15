from models import Actor, Critic
import torch
from env import CartPoleEnv
from rollout import RolloutBuffer, collect_rollout

def test_sanity_check():
    print("--- Starting Sanity Check ---")

    # 1. Setup Phase
    device = torch.device("cpu")
    env = CartPoleEnv(device=device)

    # Initialize separate networks
    actor = Actor(env.observation_space, env.action_space).to(device)
    critic = Critic(env.observation_space).to(device)

    buffer = RolloutBuffer()

    # 2. Execution Phase (Simulate one PPO iteration)
    steps_to_collect = 128
    batch_size = 32

    print(f"1. Collecting {steps_to_collect} steps of data...")
    collect_rollout(env, actor, critic, buffer, steps_to_collect)

    # Check if buffer is full
    current_len = len(buffer.states)
    if current_len != steps_to_collect:
        print(f"Buffer filled correctly: {current_len}/{steps_to_collect}")
    else:
        print(f"Buffer size mismatch: {current_len}")

    # 3. Data Processing Phase (GAE)
    print("2. Verifying GAE Computation...")
    if buffer.advantages is not None and buffer.returns is not None:
        print("GAE computed successfully (Fields are not None).")
    else:
        print("GAE failed to compute inside collect_rollout.")

    # 4. Batch Generation Phase
    print(f"3. Testing Batch Generation (Batch size: {batch_size})...")

    # Get the first batch
    data_loader = buffer.get_batches(batch_size)
    states, actions, logprobs, returns, advs = next(data_loader)

    # Define expected shapes
    # States: [32, 4], Actions: [32], Returns: [32]
    print(f"   Shape Check -> States: {states.shape}, Returns: {returns.shape}")

    expected_state_shape = (batch_size, 4)
    expected_1d_shape = (batch_size,)

    if (states.shape == expected_state_shape and
            returns.shape == expected_1d_shape and
            advs.shape == expected_1d_shape):
        print("Correct Tensor shapes.")
    else:
        print("Tensor shapes are wrong.")

if __name__ == "__main__":
    test_sanity_check()


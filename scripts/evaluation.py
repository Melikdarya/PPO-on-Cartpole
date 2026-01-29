import torch
import sys
import wandb
from src.env import CartPoleEnv
from src.ppo import PPOAgent
import multiprocessing


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Weights & Biases
    # with wandb.init(project="ppo-cartpole") as run:
    with wandb.init(project="ppo-cartpole", id="1aismqee", resume="must") as run:

        config = run.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_env = CartPoleEnv(device)


        mean_scores = []

        epoch_rewards = []

        for _ in range(config.stabilization_test_times):

            # ----------> train <----------

            CartPoleAgent = PPOAgent(train_env.observation_space, train_env.action_space, device)
            epoch_reward = CartPoleAgent.train(
                train_env,
                config.actor_learning_rate,
                config.critic_learning_rate,
                config.num_epochs,
                config.minibatch_size,
                config.steps_per_rollout,
                config.entropy_bonus_coef,
                config.discount_factor,
                config.gae_lambda,
                config.epsilon
            )

            epoch_rewards.append(epoch_reward)
            #     # ----------> save <----------

            model_name = f"epsilon_{config.epsilon}_lambda_{config.gae_lambda}_steps_{config.steps_per_rollout}.pth"
            run.name = model_name.split(".pth")[0]
            #     CartPoleAgent.save_model_parameters(model_name)
            #     run.log({"model_saved": model_name})

            # # ----------> load <----------

            # newAgent = PPOAgent(train_env.observation_space, train_env.action_space, device)
            # model_name = f"epsilon_{config.epsilon}_lambda_{config.gae_lambda}_steps_{config.steps_per_rollout}.pth"
            # newAgent.load_model_from_dict(model_name)

            # ----------> test <----------

            test_env = CartPoleEnv(device)
            results = CartPoleAgent.test(
                test_env,
                config.test_episodes,
                csv_path="results/eval.csv",
                success_threshold=config.success_threshold,
                model_name=model_name
            )
            mean_scores.append(results["mean"])

        # ----------> log to wandb <----------

        rewards_tensor = torch.tensor(mean_scores, dtype=torch.float32)
        mean = float(rewards_tensor.mean().item())
        std = float(rewards_tensor.std().item())

        average_epoch_rewards = torch.tensor(epoch_rewards, dtype=torch.float32).mean(dim=0).tolist()

        table_data = [[i, reward] for i, reward in enumerate(average_epoch_rewards)]
        table = wandb.Table(data=table_data, columns=["Epoch", "Average reward in epoch"])

        run.log({
            "mean_reward": mean,
            "std_reward": std,
            # "success_rate": results["success_rate"] if results["success_rate"] is not None else 0.0,
            "epsilon": config.epsilon,
            "gae_lambda": config.gae_lambda,
            "steps_per_rollout": config.steps_per_rollout,
            "average_epoch_rewards": wandb.plot.line(table, "Epoch", "Average reward in epoch", title="Average rewards in training epochs")
        })

        print(f"Results: {results}")
        # print(f"average_epoch_rewards: {average_epoch_rewards}")
        run.finish()


sweep_configuration = {
    "program": "evaluation.py",
    "project": "ppo-cartpole",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "mean_reward"
    },
    "parameters": {
        "actor_learning_rate": {
            "value": 0.001
        },
        "critic_learning_rate": {
            "value": 0.001
        },
        "num_epochs": {
            "value": 50
        },
        "minibatch_size": {
            "value": 64
        },
        "entropy_bonus_coef": {
            "value": 0.01
        },
        "discount_factor": {
            "value": 0.99
        },
        "success_threshold": {
            "value": 450
        },
        "test_episodes": {
            "value": 20
        },
        "stabilization_test_times": {
            "value": 5
        },  

        "steps_per_rollout": {
            "values": [125, 250, 375, 500]
        },
        "epsilon": {
            "values": [0.1, 0.2, 0.3]  # Clip range ablation
        },
        "gae_lambda": {
            "values": [0.9, 0.95, 0.97]  # GAE ablation
        },

    }
    # "early_terminate": {
    #     "type": "hyperband",
    #     "min_iter": 5
    # }
}

def run_agent(sweep_id):
    wandb.agent(sweep_id=sweep_id, function=main)

if __name__ == "__main__":
    main()
    # multiprocessing.set_start_method('spawn', force=True)
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="ppo-cartpole")
    # num_agents = 4
    # processes = [multiprocessing.Process(target=run_agent, args=(sweep_id,)) for _ in range(num_agents)]
    
    # for p in processes:
    #     p.start()
    
    # for p in processes:
    #     p.join()



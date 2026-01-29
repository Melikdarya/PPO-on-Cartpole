# PPO-on-Cartpole

### Authors:
- Melika Daryaei
- Weronika Jamróz
- Bartosz Peczyński

## Overview
This repository containt the implementation of Proximal Policy Optimization on CartPole environment. 
The project focuses on training an agent with PPO using actor-critic architecture, evaluate performance and stability on CartPole. 

## Installation & Setup
To get started with this project, follow these steps:

1. Clone the repository
```shell
git clone https://github.com/Melikdarya/PPO-on-Cartpole.git
cd PPO-on-Cartpole
```
2. Set up a virtual environment
```shell
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```shell
pip install -r requirements.txt
```

## Usage
1. Train the Agent
Set up desired hyperparameters and run training script:
```shell
python3 -m scripts.train
```
2. Check out results
```shell
python3 main.py
```

## Project Structure
```
PPO-on-CartPole/
│
├── README.md           <- The top-level README for developers using this project.
│
├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── main.py             <- Code to visualise the results and run model inference with trained models 
│
├── src
│   ├── env             <- Environment wrapper.
│   ├── models          <- Actor and Critic neural networks.
│   ├── rollout         <- RolloutBuffer, GAE, rollout collection.
│   ├── sanity check    <- Sanity tests for rollout + buffer.
│   └── ppo             <- PPO training loop + Agent class
│
└──  scripts
    ├── train           <- Train and save policy.
    ├── evaluation      <- Fine-tuning.
    └── record          <- Make videos of trained agent in action.

```

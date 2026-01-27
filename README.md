# PPO-on-Cartpole

## TODO
- [ ] Better README.md
- [ ] final report
- [ ] slides
- [ ] validation in training loop for e.g. each 10th epoch
- [ ] is requirements.txt complete?

## Project Structure
```
PPO-on-CartPole/
│
├── env.py                # Environment wrapper (CartPoleEnv)
├── models.py             # Actor and Critic neural networks
├── rollout.py            # RolloutBuffer, GAE, rollout collection
│
├── sanity_check.py       # Sanity tests for rollout + buffer
│
├── train.py              # (TO DO) PPO training loop
├── evaluate.py           # (TO DO) Policy evaluation & rendering
│
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

## How the Components Work Together
```
env.py
↓
rollout.py  ←  models.py (Actor & Critic)
↓
RolloutBuffer + GAE
↓
train.py (PPO update)
↓
evaluate.py
```

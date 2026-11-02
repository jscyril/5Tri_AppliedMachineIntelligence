# Lab 1: Q-Learning

This repository contains `Q_Learning.ipynb`, which implements and analyzes a tabular Q-learning agent using an epsilon-greedy policy in Gymnasium's deterministic `FrozenLake-v1` environment.

## Notebook contents

- Trains with 500, 1,000, 2,000, 5,000, and 10,000 episodes.
- Starts epsilon at 1.0 and decreases it during training.
- Records reward for every training episode.
- Plots cumulative reward and the rolling average reward over the latest 100 episodes.
- Tests each learned policy for 100 episodes with exploration disabled.
- Compares success rates in a summary table and bar chart.
- Displays the environment and the extracted greedy policy.

## Run

Install the required packages, then open and run the notebook:

```bash
pip install gymnasium numpy pandas matplotlib jupyter
jupyter notebook Q_Learning.ipynb
```

The notebook is on the `lab01` branch; `main` is kept as the separate base branch.

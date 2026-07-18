# Neural maximization-bias results

The metric is the probability of selecting the biased left action, so lower
values are better. Results are means over 1,000 independent iterations.

| Algorithm | Final probability (95% CI) | Last-50 mean | Curve AUC |
| --- | ---: | ---: | ---: |
| DQN | 0.07124 ± 0.01371 | 0.07121 | 0.07815 |
| SOR DQN, w=1.3 | 0.07459 ± 0.01456 | 0.07471 | 0.08088 |
| Double DQN | 0.03386 ± 0.01015 | 0.03394 | 0.04048 |
| Double SOR DQN, w=1.3 | **0.01098 ± 0.00546** | **0.01118** | **0.02083** |

Relative to Double DQN, Double SOR DQN reduces the final biased-action
probability by 67.56%, the last-50-episode mean by 67.05%, and the curve AUC by
48.55%.

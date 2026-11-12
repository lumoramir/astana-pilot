# Training notes: what went wrong once, and how it was found

A useful story for the committee: the first overnight run (`night1`, 8 h) ended with **0 % success and 99.7 % wall collisions**.
It was not "too little training" - the curve was flat from the first step. The diagnosis, step by step:

| Observation | Measurement | Cause | Fix |
|---|---|---|---|
| Clone drives, PPO result does not | cloned policy (`bc_model.zip`): 61 % of the route; `final_model.zip`: crash within ~2 s | PPO destroyed it | see below |
| Clone crashes with exploration noise | deterministic: 300-1900 steps; with gSDE noise `log_std=-2`: 10-60 steps; `log_std=-4`: ~300 steps | exploration noise too strong for a cloned policy | `PPOConfig.finetune().log_std_init = -4` |
| Still collapsed with weaker noise (`fixcheck2`: success 0.18 -> 0.00, KL 0.33 on the first update) | the critic was fitted on *expert* demos, so on the learner's own states its advantages are noise | PPO's first updates tear the policy | `CriticWarmupCallback`: policy frozen for 6 PPO iterations, only the critic learns |
| With warm-up (`fixcheck3`) | success 0.43 -> 0.65 -> 0.79 in 226 k steps, 625 steps/s | - | kept |
| Expert itself finished only 33 % of trips | 12 of 24 trips ended in a time-out, often standing behind a car frozen at the end of *its* route | simulator bug: a car without a route continuation froze for ever | cars leave after 6 s; retry with cool-down |
| Expert hit pedestrians with the car's side | it judged "will I pass before the person arrives" by the front bumper only | expert logic | uses the whole body length (`t_rear`) |
| Endless waiting at zebras | a continuous stream of pedestrians kept the crossing occupied | world model | one group crosses, then 12 s for the cars |
| People walked into a standing car | no pedestrian-vehicle avoidance | world model | people wait (max 5 s) instead of stepping into a standing car's body |

Result of the simulator fixes alone: the rule-based expert on EXPO at full difficulty went from 8/24 to 17-19/24 completed trips (small
sample, same seeds).

Method: never trust a long run that has not been checked for one minute. Plot early (`astana plot <run> --watch`), look at
`custom/progress` (share of the route driven) - it moves long before `custom/success` does.

Regression tests: `tests/test_training_guards.py`.

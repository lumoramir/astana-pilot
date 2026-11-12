# 🤖 Published network: `astana_pilot.zip`

The trained driving policy that ships with the repository (Stable-Baselines3 PPO, 8.7 MB). Everything else that training produces
(checkpoints, TensorBoard logs, expert demonstrations — ~0.5 GB) stays out of git (`runs/` in `.gitignore`).

| | |
|---|---|
| Origin | `runs/night3/best_model.zip` — the best network of the fine-tuning run `night3` |
| Training | behaviour cloning + DAgger on expert demonstrations → PPO (`night2`, 0.16 M steps) → fine-tuning with rule-violation penalties × 2.5 (`night3`, selected at 0.88 M steps by the stability guard; the run went on to 5.1 M steps / 6.4 h) |
| Cities in training | Astana EXPO, Astana Nurzhol (OpenStreetMap), synthetic:0, synthetic:1 |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 12 parallel simulators |
| Inputs / outputs | 351 numbers (lidar, own state, next ~32 m of the route, signals / signs / crossings, speed limit, road level, 8 nearest road users; **no coordinates**) → steering, pedal |
| SHA-256 | `C682E86BC19ED1DAD65C0A166263C85485C212DE3CED28F9D122EC9D3F4236A6` |

**Measured** (30 paired trips per group, full traffic): trips completed 56.7 % seen / 53.3 % unseen (expert 60.0 % / 56.7 %),
trips with a rule violation 36.7 % / 23.3 % (expert 0 % / 3.3 %). Full report: [`eval_report.md`](eval_report.md).

```bat
astana view --map assets\maps\astana_expo.pkl --model demo_weights\astana_pilot.zip --pick
astana eval --model demo_weights\astana_pilot.zip --episodes 60 --seen-maps assets\maps\astana_expo.pkl,assets\maps\astana_nurzhol.pkl,synthetic:0,synthetic:1 --unseen-maps synthetic:100,synthetic:101,synthetic:102,synthetic:103
```

Research prototype for simulation only — not a controller for a real vehicle.

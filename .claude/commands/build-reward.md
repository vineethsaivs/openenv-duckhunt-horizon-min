Build the GRPO reward function in training/reward.py.

Five signals:
1. Hit: +1.0, double_kill: +2.5
2. Horizon penalty on hits: -0.1 * (horizon/30)
3. Miss: -0.3 + proximity_bonus where proximity_bonus = 0.5 * exp(-5.0 * distance)
4. No target: -0.5
5. Invalid format: -1.0

The regex must match: shoot(x=0.42, y=0.31, horizon=3)
Pattern: r'shoot\(x=([\d.]+),\s*y=([\d.]+),\s*horizon=(\d+)\)'

Also create training/prompts.py with the system prompt for the VLM.

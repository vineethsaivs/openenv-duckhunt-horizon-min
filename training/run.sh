#!/bin/bash
cd ~/openenv-duckhunt-horizon-min
python training/train_grpo.py --max-steps 500 --output-dir outputs/duckhunt_grpo

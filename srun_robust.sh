#/bin/bash
srun --time 3-0 --qos=32gpu7d  --gres=gpu:1 python main.py --robust --lr 0.0005&

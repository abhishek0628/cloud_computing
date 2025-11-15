#!/usr/bin/env bash
python -m src.trainer --epochs 120 --num_ants 24 --num_colonies 6 --seed 42
python -m src.eval --model experiments/best_model.pth --out_dir out

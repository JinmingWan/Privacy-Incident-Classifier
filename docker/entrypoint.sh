#!/usr/bin/env bash
set -e
python src/preprocess.py
python src/train.py
python src/evaluate.py

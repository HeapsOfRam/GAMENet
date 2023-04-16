#!/bin/bash

MODEL_NAME=GAMENet

echo "PREPARING DATA..."
python ../data/prepare_data/prepare_data.py
echo "CONSTRUCTING ADJACENCY MATRICES"
python ../data/prepare_data/construct_adj.py
echo "TRAINING MODEL..."
python train_GAMENet.py --model_name $MODEL_NAME --ddi
echo "MODEL EVALUATION..."
python train_GAMENet.py --model_name $MODEL_NAME --ddi --resume_path ./saved/$MODEL_NAME/final.model --eval
## baselines
echo "BASELINE COMPARISON..."
echo "NEAREST BASELINE..."
echo "train..."
python baseline/baseline_near.py "false"
echo "eval..."
python baseline/baseline_near.py "true"
echo "DMNC BASELINE..."
echo "train..."
python baseline/train_DMNC.py "false"
echo "eval..."
python baseline/train_DMNC.py "true"
echo "LEAP BASELINE..."
echo "train..."
python baseline/train_Leap.py "false"
echo "eval..."
python baseline/train_Leap.py "true"
echo "LR BASELINE..."
echo "train..."
python baseline/train_LR.py "false"
echo "eval..."
python baseline/train_LR.py "true"
echo "RETAIN BASELINE..."
echo "train..."
python baseline/train_Retain.py "false"
echo "eval..."
python baseline/train_Retain.py "true"

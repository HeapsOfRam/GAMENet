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

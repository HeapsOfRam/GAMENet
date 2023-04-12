#!/bin/bash

MODEL_NAME=GAMENet

python train_GAMENet.py --model_name $MODEL_NAME --ddi
python train_GAMENet.py --model_name $MODEL_NAME --ddi --resume_path ./saved/$MODEL_NAME/final.model --eval

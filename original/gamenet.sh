#!/bin/bash

MODEL_NAME=GAMENet

# set flags to determine whether to prepare the data
should_prep_data=true

# set flags to determine whether to run the various baselines
run_near=true
run_dmnc=false
run_leap=true
run_lr=true
run_retain=true
skip_baselines=false

# set flags to determine whether to run gamenet
should_train_gn=true
should_eval_gn=true

prepare_data() {
    echo "deleting old pkl files..."
    rm data/pkl/*.pkl
    echo "PREPARING DATA..."
    python prepare_data.py
    echo "CONSTRUCTING ADJACENCY MATRICES"
    python construct_adj.py
}

train_gamenet() {
    echo "TRAINING GAMENET MODEL..."
    python train_GAMENet.py --model_name $MODEL_NAME --ddi
}

eval_gamenet() {
    echo "GAMENET MODEL EVALUATION..."
    python train_GAMENet.py --model_name $MODEL_NAME --ddi --resume_path ./saved/$MODEL_NAME/final.model --eval
}

## baselines
train_near() {
    echo "NEAREST BASELINE train..."
    python baseline_near.py "false"
}

train_dmnc() {
    echo "DMNC BASELINE train..."
    python train_DMNC.py "false"
}

train_leap() {
    echo "LEAP BASELINE train..."
    python train_Leap.py "false"
}

train_lr() {
    echo "LR BASELINE train ..."
    python train_LR.py "false"
}

train_retain() {
    echo "RETAIN BASELINE train..."
    python train_Retain.py "false"
}

train_baselines() {
    run_near=$1
    run_dmnc=$2
    run_leap=$3
    run_lr=$4
    run_retain=$5

    if [ "$run_near" = true ] ; then
        train_near
    else
        echo "skipping near train..."
    fi
    if [ "$run_dmnc" = true ] ; then
        train_dmnc
    else
        echo "skipping dmnc train..."
    fi
    if [ "$run_leap" = true ] ; then
        train_leap
    else
        echo "skipping leap train..."
    fi
    if [ "$run_lr" = true ] ; then
        train_lr
    else
        echo "skipping lr train..."
    fi
    if [ "$run_retain" = true ] ; then
        train_retain
    else
        echo "skipping retain train..."
    fi
}


eval_near() {
    echo "NEAREST BASELINE eval..."
    python baseline_near.py "true"
}

eval_dmnc() {
    echo "DMNC BASELINE eval..."
    python train_DMNC.py "true"
}

eval_leap() {
    echo "LEAP BASELINE eval..."
    python train_Leap.py "true"
}

eval_lr() {
    echo "LR BASELINE eval ..."
    python train_LR.py "true"
}

eval_retain() {
    echo "RETAIN BASELINE eval..."
    python train_Retain.py "true"
}

eval_baselines() {
    run_near=$1
    run_dmnc=$2
    run_leap=$3
    run_lr=$4
    run_retain=$5

    if [ "$run_near" = true ] ; then
        eval_near
    else
        echo "skipping near eval..."
    fi
    if [ "$run_dmnc" = true ] ; then
        eval_dmnc
    else
        echo "skipping dmnc eval..."
    fi
    if [ "$run_leap" = true ] ; then
        eval_leap
    else
        echo "skipping leap eval..."
    fi
    if [ "$run_lr" = true ] ; then
        eval_lr
    else
        echo "skipping lr eval..."
    fi
    if [ "$run_retain" = true ] ; then
        eval_retain
    else
        echo "skipping retain eval..."
    fi
}

# prepare data
if [ "$should_prep_data" = true ] ; then
    prepare_data
else
    echo "skipping data prep..."
fi

# train gamenet
if [ "$should_train_gn" = true ] ; then
    train_gamenet
else
    echo "skipping gamenet train..."
fi

# train and eval baselines
if [ "$skip_baselines" = true ] ; then
    echo "skipping all baselines..."
else
    train_baselines $run_near $run_dmnc $run_leap $run_lr $run_retain
    eval_baselines $run_near $run_dmnc $run_leap $run_lr $run_retain
fi

# eval gamenet
if [ "$should_eval_gn" = true ] ; then
    eval_gamenet
else
    echo "skipping gamenet eval..."
fi

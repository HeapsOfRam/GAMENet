# GAMENet: Reproduction Study

I have forked [the original code from the GAMENet paper](https://github.com/sjy1203/GAMENet) here.
So far, all I have really done is modify the logic presented there to run in containers.
Now, the data preparation, the baseline model training and evaluation, and the GAMENet model training and evaluation all run within the specified container.
It may make sense to modify this so that I run these different logics in parallel in different containers, to get the most efficient utilization of my GPU.
For now, everything just runs as one core pipeline.

## Running the Code

As mentioned, the code has now been containerized, so it should work regardless of the system you are running on.
I used [Pod Manager tool (podman)](https://podman.io/) to run these containers.
Podman is a drop-in replacement for [the Docker tool](https://www.docker.com/) for most use-cases in my experience.
So, the files I provide should work with either `podman` or with `docker`.
However, you will need to install and enable [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime) in order to access the GPU resources on the host system.
For me, following the [NVIDIA Container Runtime installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) were sufficient to get this working with both Docker and Podman.

Once the NVIDIA Container Runtime has been installed, the code should run via the [`build_and_run.sh` script](./build_and_run.sh).
Usage is as follows:

```bash
# can run the provided script
# must give version as a command line argument
sh build_and_run.sh $VERSION
# for example, can run:
sh build_and_run v01
```

However, the [`build_and_run.sh` script](./build_and_run.sh) mainly just has the two commands to build and then run the container.
These commands are as follows:

```bash
VERS=$DESIRED_VERSION # ie, v01
IMAGE=$DESIRED_IMAGE_NAME # ie, ryangamenet

podman build -t $IMAGE:$VERS .
# note i need to use --privileged mode to access the gpus, i haven't figured out a workaround
# also, provide the --gpus all argument here
podman run --privileged --gpus all -it --rm $IMAGE:$VERS
```

Please not above the need for both the `--privileged` flag as well as the `--gpus all` argument.
It seems there may be some alternative ways to run this from the NVIDIA Container Runtime docs, but this is what worked best for me.

### Flags

Right now, my approach does not have the flexibility of the original code.
In other words, I cannot simply choose to run a baseline, or the GAMENet model, or skip the data preparation, etc.
Instead, the workaround is to manipulate the flags within the [`gamenet.sh` script](./gamenet.sh), which is the main entrypoint of the container logic.
Specifically, notice these variables, which can be toggled between either `true` or `false`:

```bash
# set flags to determine whether to prepare the data
should_prep_data=true

# set flags to determine whether to run the various baselines
run_near=true
run_dmnc=false
run_leap=false
run_lr=true
run_retain=false
skip_baselines=false

# set flags to determine whether to run gamenet
should_train_gn=true
should_eval_gn=true
```

The above show the default values I have for now.
I've set the DMNC, RETAIN, and Leap model training/evaluation to false for now as they take longer to run.
However, I have confirmed that they do run when their flags are set to `true`.
One can also set the `gn` flags to `false` if they do not want to run the GAMENet model.

If the `should_prep_data` flag is set to `false`, then the process will use the pickle files as provided in the original repository.
If this flag is set to `true`, then you will need to download the DDI data and the MIMIC dataset and extract them into the directory [`data/data/`](./data/data/).
The following files are needed:

- `drug-DDI.csv`: DDI graph; a download link is included in the original README
- `DIAGNOSES_ICD.csv`: from MIMIC data
- `PRESCRIPTIONS.csv`: from MIMIC data
- `PROCEDURES_ICD.csv`: from MIMIC data

So far, the project uses MIMIC-III data.
MIMIC-IV is planned as a future enhancement to my replication study.

### Cleanup

Finally, sometimes it is helpful to clean up the podman environment.
For some reason, I need to force some resources to stop before I am able to `prune` to free up all resources.
These commands have been organized in the [`cleanup.sh` script](./cleanup.sh).

## Cite 

The authors have asked to cite their paper when using their work:

```
@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}
```

---

Below is the original README from [the original repository](https://github.com/sjy1203/GAMENet).

# ORIGINAL README

# GAMENet

GAMENet : Graph Augmented MEmory Networks for Recommending Medication Combination

For reproduction of medication prediction results in our [paper](https://arxiv.org/abs/1809.01852), see instructions below.

## Overview
This repository contains code necessary to run GAMENet model. GAMENet is an end-to-end model mainly based on graph convolutional networks (GCN) and memory augmented nerual networks (MANN). Paitent history information and drug-drug interactions knowledge are utilized to provide safe and personalized recommendation of medication combination. GAMENet is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/) and outperformed several state-of-the-art deep learning methods in heathcare area in all effectiveness measures and also achieved higher DDI rate reduction from existing EHR data.


## Requirements
- Pytorch >=0.4
- Python >=3.5


## Running the code
### Data preprocessing
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:
1.  download [MIMIC data](https://mimic.physionet.org/gettingstarted/dbsetup/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
2.  download [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) and put it in ./data/
3.  run code **./data/EDA.ipynb**

Data information in ./data:
  - records_final.pkl is the input data with four dimension (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  - voc_final.pkl is the vocabulary list to transform medical word to corresponding idx.
  - ddi_A_final.pkl and ehr_adj_final.pkl are drug-drug adjacency matrix constructed from EHR and DDI dataset.
  - drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt are mapping files for drug code transformation.
  
  
### Model Comparation
 Traning codes can be found in ./code/baseline/
 
 - **Nearest** will simply recommend the same combination medications at previous visit for current visit.
 - **Logistic Regression (LR)** is a logistic regression with L2 regularization. Here we represent the input data by sum of one-hot vector. Binary relevance technique is used to handle multi-label output.
 - **Leap** is an instance-based medication combination recommendation method.
 - **RETAIN** can provide sequential prediction of medication combination based on a two-level neural attention model that detects influential past visits and significant clinical variables within those visits.
 - **DMNC** is a recent work of medication combination prediction via memory augmented neural network based on differentiable neural computers. 
 
 
 ### GAMENet
 ```
 python train_GAMENet.py --model_name GAMENet --ddi# training with DDI knowledge
 python train_GAMENet.py --model_name GAMENet --ddi --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 python train_GAMENet.py --model_name GAMENet # training without DDI knowledge
 python train_GAMENet.py --model_name GAMENet --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 ```
 
## Cite 

Please cite our paper if you use this code in your own work:

```
@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}
```

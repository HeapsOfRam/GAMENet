# GAMENet: Reproduction Study in PyHealth

GAMENet is already implemented via the [`pyhealth` library](https://pyhealth.readthedocs.io/en/develop/api/models/pyhealth.models.GAMENet.html).
This codebase serves to run that training and evaluation pipeline, containerize it, and add several ablations that should solidify some of the conclusions ascertained in the original paper.

[The gamenet.py script](./gamenet.py) serves as the main entrypoint for the container/logic.
It accepts several flags, as documented below.
This script can be run in an environment with the right dependencies, locally, or as a container.
Please see [the `build_and_run.sh` script](./build_and_run.sh) for details for how to run as a container.

## Background

GAMENet is a model for recommending safe drug combinations to prescribe to patients.
Patient procedures and conditions are input, fed into embeddings networks and then passed into Gated Recurrent Units (GRU), which forms the basis of a patient representation.
This patient representation is used as a query against a Dynamic Memory component which contains patient medication history.
A Memory Bank (Static Memory) component is included as well.
The Memory Bank sums an EHR Drug Graph together with a DDI Graph.
The EHR Drug Graph shows drugs that are commonly prescribed together within the EHR dataset, and acts as a positive signal.
The DDI Graph shows drugs that have interactions when prescribed together, and acts as a negative signal.
Then, the patient query, the output from the Memory Bank, and the output from the Dynamic Memory are concatenated and fed into a Sigmoid activation function to produce predictions for the current visit.

## Running the Code

### Prerequisites

#### NVIDIA Container Runtime

If you want to run the logic in containers, you will need to install and enable [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime) in order to access the GPU resources on the host system.
For me, following the [NVIDIA Container Runtime installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) were sufficient to get this working with both Docker and Podman.

#### Data

To run the code, you will first need either the MIMIC3 or MIMIC4 data depending on which pipeline you would like to run.
The data can be downloaded from [the PhysioNet site](https://mimic.physionet.org/).
Once downloaded, please extract and place in the following directory structure:
```bash
# please note, these paths are relative to this readme file's location in the repository
# for mimic3 data
./hiddendata/extracted/mimic3/
# for mimic4 data
./hiddendata/extracted/mimic4/
```

The code expects the above to be the root path for the MIMIC datasets, though this can be changed by modifying the `_DATA_DIR` variable in [the main entrypoint python script](./gamenet.py).

#### Dependencies

Dependencies are listed in the [requirements.txt file](./requirements.txt).
Steps for installing these into a virtual environment are shown below.
The Dockerfile includes a step to install these repositories.
Additionally, I am using the following version of python:

```bash
$ python --version
Python 3.8.16
```

### Running

Training and evaluation are both run as part of the same pipeline.
This can be found either in [the gamenet.py entrypoint](./gamenet.py) or in [the EDA notebook](./EDA.ipynb).

#### Podman/Docker

The code runs in podman containers, similar to the original replication work performed.
The code can be run either locally or via a container.
[The `build_and_run.sh` script](./build_and_run.sh) demonstrates how to run the logic in a container format.
Its contents are as follows:

```bash
# first, build the image with a given image name and version tag
docker build -t $IMAGE:$VERS .
# then, run the image with gpu access
# make sure to bind the data directory with the mimic3/mimic4 data using the `--mount` flag
# can optionally pass in arguments to the container that will be passed to the python script
## for an example of this, see the `--mimic` flag below, which is an argument to the python script
docker run --privileged --gpus all -it --rm --mount type=bind,source="$(pwd)"/hiddendata,target=/app/hiddendata/ $IMAGE:$VERS --mimic=$MIMIC
```

#### Running Locally

To run locally, you can simply run the python script directly:

```bash
# first, make sure you are in the pyhealth directory as your root
cd pyhealth
# make sure you have a virtual environment that can support the code, ie:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# then, simply run the script
python gamenet.py
```

#### Flags

[The main python entrypoint](./gamenet.py) takes in a number of optional arguments:

 - `-m`, `--mimic`: determines whether to use MIMIC3 or MIMIC4 data. Possible values are `3` and `4`. The default is `4`
 - `-s`, `--skip-baseline`: determines whether or not to run the RETAIN baseline. Adding this argument skips the baseline training and evaluation
 - `-b`, `--baseline-only`: determines whether or not to run the GAMENet model. Adding this argument skips the GAMENet training and evaluation, and will only run the baselines (unless `-s` is also provided, in which case nothing will happen)
 - `-e`, `--epochs`: number of epochs to run training for (for both baseline and GAMENet models). The default is `20`
 - `-d`, `--decay-weight`: sets the decay weight for training (for both baseline and GAMENet models). The default is `1e-5`
 - `-l`, `--learning-rate`: sets the learning rate for training (for both baseline and GAMENet models). The default is `1e-3`
 - `-t`, `--task`: adds a task to the list of tasks to be valuated (for both baseline and GAMENet models). This changes the data preparation as well as some of the models run. Right now, just three tasks are allowed: `"drug_recommendation"`, `"no_hist"`, and `"no_proc"`. `"no_hist"` prepares the data without accounting for the patient's history, `"no_proc"` prepares the data (and the models) without accounting for any procedure codes, and `"drug_recommendation"` is the default pyhealth-provided task. `"no_hist"` and `"no_proc"` are currently only supported for MIMIC4 data. This argument can be passed multiple times, with different tasks as input
 - `-a`, `--all-tasks`: passing this argument will run all of the available tasks for the given dataset (either MIMIC3 or MIMIC4). This argument supersedes the `-t` argument
 - `--dev`: whether to read the MIMIC data in "dev" mode. This reads only in samples of the MIMIC data, making evaluation much quicker -- however, the results will not be quite as accurate

All of these flags have default values and do not have to be manually provided.

### Ablations

There were two ablations that I focused on for this work.

#### History Ablation

For this ablation, I omitted patient history information in the data preparation step.
The RETAIN model was able to handle this data as is.
For GAMENet, I needed to do make some modifications.
I removed the Dynamic Memory component from the architecture to make this ablation work properly.
This ablation can be run with the `"no_hist"` task, which includes both the data preparation differences as well as the model differences.

#### Procedure Ablation

For this ablation, I omitted patient procedure information in the data preparation step.
I needed to also remove the reference to this as a feature when building the RETAIN model.
In GAMENet, I removed the Gated Recurrent Unit (GRU) corresponding to the procedure information, and modified the `forward` function to not expect procedure information.
I also had to remove the processing related to procedures in the architecture.
This ablation can be run with the `"no_proc"` task, which includes both the data preparation differences as well as the model differences.

### Tasks

There are 3 tasks that can be run.
All tasks are created for the MIMIC dataset.
Primarily, I focused on MIMIC-IV, so only the default task is available in MIMIC-III.

 - `"drug_recommendation"`: this is the default drug recommendation task [from `pyhealth`](https://pyhealth.readthedocs.io/en/latest/api/tasks/pyhealth.tasks.drug_recommendation.html)
 - `"no_hist"`: this corresponds to the history ablation, and removes the patient history from data processing. It also removes the Dynamic Memory component from the GAMENet architecture.
 - `"no_proc"`: this corresponds to the procedure ablation, and removes patient procedures from the data processing. It also removes the procedure GRU from the GAMENet architecture.

Tasks (and the corresponding models) are processed in isolation.
In other words, the `"no_hist"` task makes an instance of GAMENet without the Dynamic Memory component, but other tasks will run with this component (unless explicitly specified otherwise, see [`MODEL_TYPES_PER_TASK` variable in the constants.py file](./constants.py)).

### EDA Notebook

I've written a notebook showing how to load in the data and train/evaluate some models based on my main logic [in EDA.ipynb](./EDA.ipynb).
See [this old version](https://github.com/HeapsOfRam/GAMENet/blob/5fbb96e549e5aaf9ec2e4ffeef1bff06d0589e67/pyhealth/EDA.ipynb) which provided the results I ended up using in my report.

### Cleanup

#### Local

It can be helpful to remove the previous models before a run.
For this, can simply run:

```sh
rm -r output/
```

#### Podman

Finally, sometimes it is helpful to clean up the podman environment.
For some reason, I need to force some resources to stop before I am able to `prune` to free up all resources.
These commands have been organized in the [`cleanup.sh` script](./cleanup.sh).

## Results

Generally, results for various runs can be found [in this directory](./results/).
Specifically, see the [results of my final run here](./results/final_results.txt).
These results were generated with the model weights that are saved [in the `output` directory](./output/).
I put the above results into a table here:


| Model   | Task                | Accuracy              | Jaccard Score       | Precision          | Recall              | PR AUC           | F1                 | Avg DPV            | DDI Rate             | Delta DDI Rate         |
|---------|---------------------|-----------------------|---------------------|--------------------|---------------------|------------------|--------------------|--------------------|----------------------|------------------------|
| RETAIN  | drug_recommendation | 0.0021432522123893804 | 0.41576684941022884 | 0.7603958754470037 | 0.49021940550346643 | 0.72476353605    | 0.5672969181604842 | 14.045423119469026 | 0.0654986379636298   | -0.012201362036370206  |
| RETAIN  | no_hist             | 0.0020545130803999454 | 0.4403503103715233  | 0.7692379557407789 | 0.5197915818336155  | 0.7495310957313  | 0.5918108028574632 | 15.063484454184358 | 0.06440503924167078  | -0.013294960758329227  |
| RETAIN  | no_proc             | 0.0020545130803999454 | 0.39668790639328144 | 0.7536945860602416 | 0.469864026383963   | 0.7093126534065  | 0.550001573210491  | 11.887925673945645 | 0.08184647568115476  | 0.004146475681154754   |
| GAMENet | drug_recommendation | 0.0015210176991150442 | 0.44669104646620933 | 0.7459566489183552 | 0.5407909737332947  | 0.741756464946   | 0.5988409450140139 | 15.948423672566372 | 0.06999259477481488  | -0.007707405225185121  |
| GAMENet | no_hist             | 0.0028763183125599234 | 0.462600997469886   | 0.7531064780331098 | 0.5629392744174778  | 0.75796644328869 | 0.6127631391726239 | 17.326599096014245 | 0.061112876787601925 | -0.01658712321239808   |
| GAMENet | no_proc             | 0.0008047112184059402 | 0.4163686653728236  | 0.7404166023417065 | 0.5055376658406298  | 0.7171681505122  | 0.569333171881927  | 13.442956947949815 | 0.075309192161028    | -0.0023908078389720117 |


Avg DPV is average number of drugs recommended per visit.
DDI Rate is calculated by:

$$\text{DDI Rate} = \frac{\sum^N_{k}\sum^{T_k}_t\sum_{i,j}\text{\textbar}\{(c_i, c_j) \in \hat{Y}^{(k)}_t \text{\textbar} (c_i, c_j) \in E_d\} \text{\textbar}}{\sum^N_k\sum^{T_k}_t\sum_{i,j}1}$$

Where `N` is the number of patients in the test set.
`T_k` is the number of visits for patient `k`.
`(c_i,c_j)` is a medication pair in recommendation set `Y_hat`, but is only counted when the pair has an edge set in `E_d` in the DDI graph.
The numerator is then normalized by the sum of all the medication combinations for every visit for every patient in `T`.

Delta DDI Rate is the change in DDI Rate from the baseline, which is considered in the original paper to be 0.0777.

Interestingly, RETAIN performs quite well here, as does the history ablation.
The history ablation has a better DDI Rate for GAMENet despite recommending more drugs.
The accuracy score even increases here.
However, the Jaccard score does go down.
It seems RETAIN did have a better DDI Rate in this run, however.

## Video

I recorded a brief video for this assignment as well.
The video goes over the purpose and implementation of GAMENet, as well as some of the results of my reproduction study.
That video [can be viewed on YouTube here](https://youtu.be/6ZBUOQaIBhQ).

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

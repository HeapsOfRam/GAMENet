# GAMENet: Reproduction Study in PyHealth

GAMENet is already implemented via the [`pyhealth` library](https://pyhealth.readthedocs.io/en/develop/api/models/pyhealth.models.GAMENet.html).
This codebase serves to run that training and evaluation pipeline, containerize it, and add several ablations that should solidify some of the conclusions ascertained in the original paper.

[The gamenet.py script](./gamenet.py) serves as the main entrypoint for the container/logic.
It accepts several flags, as documented below.
This script can be run in an environment with the right dependencies, locally, or as a container.
Please see [the `build_and_run.sh` script](./build_and_run.sh) for details for how to run as a container.

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

### Running

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
python3 gamenet.py
```

#### Flags

[The main python entrypoint](./gamenet.py) takes in a number of optional arguments:

 - `-m`, `--mimic`: determines whether to use MIMIC3 or MIMIC4 data. Possible values are `3` and `4`. The default is `4`
 - `-s`, `--skip-baseline`: determines whether or not to run the RETAIN baseline. Adding this argument skips the baseline training and evaluation
 - `-b`, `--baseline-only`: determines whether or not to run the GAMENet model. Adding this argument skips the GAMENet training and evaluation, and will only run the baselines (unless `-s` is also provided, in which case nothing will happen)
 - `-e`, `--epochs`: number of epochs to run training for (for both baseline and GAMENet models). The default is `20`
 - `-d`, `--decay-weight`: sets the decay weight for training (for both baseline and GAMENet models). The default is `1e-5`
 - `-l`, `--learning-rate`: sets the learning rate for training (for both baseline and GAMENet models). The default is `1e-3`

All of these flags have default values and do not have to be manually provided.

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

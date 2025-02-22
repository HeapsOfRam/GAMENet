# GAMENet: Reproduction Study

I have forked [the original code from the GAMENet paper](https://github.com/sjy1203/GAMENet) here.
Originally, I simply containerized the initial logic and used that to reproduce results for the GAMENet model and baselines.
Now, I have made a [separate directory called `pyhealth`](./pyhealth/) that trains and evaluates the GAMENet model and one baseline (RETAIN) within the pyhealth framework.
This implementation should make certain tasks, like my planned ablations, a bit easier to perform.

---

## PyHealth

GAMENet is already implemented via the [`pyhealth` library](https://pyhealth.readthedocs.io/en/develop/api/models/pyhealth.models.GAMENet.html).
Please see [the `pyhealth` directory](./pyhealth/) for further details.
I plan to update both this `README` and [the `README` in the `pyhealth` directory](./pyhealth/README.md) as I update the code.
However, in the event that there are conflicts between the two `README`s, please assume that [the `README` in the `pyhealth` directory](./pyhealth/README.md) is more up-to-date and thus accurate.
Please refer to the `README` in the `pyhealth` directory for more details and complete documentation on that approach.

### Running the Code

#### Data

To run the code, you will first need either the MIMIC3 or MIMIC4 data depending on which pipeline you would like to run.
The data can be downloaded from [the PhysioNet site](https://mimic.physionet.org/).
Once downloaded, please extract and place in the following directory structure:
```bash
# for mimic3 data
./pyhealth/hiddendata/extracted/mimic3/
# for mimic4 data
./pyhealth/hiddendata/extracted/mimic4/
```

The code expects the above to be the root path for the MIMIC datasets, though this can be changed by modifying the `_DATA_DIR` variable in [the main entrypoint python script](./pyhealth/gamenet.py).

#### Running

The code runs in podman containers, similar to the original replication work performed.
The code can be run either locally or via a container.
[The `build_and_run.sh` script](./pyhealth/build_and_run.sh) demonstrates how to run the logic in a container format.
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

Flags can also be provided to the `gamenet` script.
For a more complete description, please see [the `README` in the pyhealth directory](./pyhealth/README.md).

---

## Original

### Running the Code

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

However, the [`build_and_run.sh` script](./original/build_and_run.sh) mainly just has the two commands to build and then run the container.
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

#### Flags

Right now, my approach does not have the flexibility of the original code.
In other words, I cannot simply choose to run a baseline, or the GAMENet model, or skip the data preparation, etc.
Instead, the workaround is to manipulate the flags within the [`gamenet.sh` script](./original/gamenet.sh), which is the main entrypoint of the container logic.
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
If this flag is set to `true`, then you will need to download the DDI data and the MIMIC dataset and extract them into the directory [`data/data/`](./original/data/data/).
The following files are needed:

- `drug-DDI.csv`: DDI graph; a download link is included in the [original README](./original/README.md)
- `DIAGNOSES_ICD.csv`: from MIMIC data
- `PRESCRIPTIONS.csv`: from MIMIC data
- `PROCEDURES_ICD.csv`: from MIMIC data

So far, the project uses MIMIC-III data.
MIMIC-IV is planned as a future enhancement to my replication study.

#### Cleanup

Finally, sometimes it is helpful to clean up the podman environment.
For some reason, I need to force some resources to stop before I am able to `prune` to free up all resources.
These commands have been organized in the [`cleanup.sh` script](./original/cleanup.sh).

---

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

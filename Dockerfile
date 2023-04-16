#FROM python:3.7.16
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

###
#RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
#    libsm6 libxext6 libxrender-dev curl \
#    && rm -rf /var/lib/apt/lists/*
#
#RUN echo "**** Installing Python ****" && \
#    add-apt-repository ppa:deadsnakes/ppa &&  \
#    apt-get install -y build-essential python3.7 python3.7-dev python3-pip && \
#    curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python3.7 get-pip.py && \
#    rm -rf /var/lib/apt/lists/*
###

###
RUN apt-get update && apt-get install -y curl wget gcc build-essential

# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# create env with python 3.7
RUN /opt/conda/bin/conda create -y -n myenv python=3.7
ENV PATH=/opt/conda/envs/myenv/bin:$PATH
###

## copy files
#COPY . /app
## copy files to the code directory for unpickling
#COPY data/prepare_data/read_functions.py /app/code/
#COPY data/prepare_data/create_vocabulary.py /app/code/
#COPY data/prepare_data/construct_adj.py /app/code/
#COPY data/prepare_data/prepare_data.py /app/code/
## copy modules to baseline directory for imports
#COPY code/util.py /app/code/baseline/
#COPY code/models.py /app/code/baseline/
#COPY code/layers.py /app/code/baseline/
## copy files to baseline directory for unpickling
#COPY data/prepare_data/read_functions.py /app/code/baseline/
#COPY data/prepare_data/create_vocabulary.py /app/code/baseline/
#COPY data/prepare_data/construct_adj.py /app/code/baseline/
#COPY data/prepare_data/prepare_data.py /app/code/baseline/
#WORKDIR /app/code/

# copy main shell script and requirements
COPY gamenet.sh /app/
COPY requirements.txt /app/
# copy python files from code directory
COPY code/layers.py /app/
COPY code/models.py /app/
COPY code/util.py /app/
COPY code/train_GAMENet.py /app/
# copy baselines
COPY code/baseline/baseline_near.py /app/
COPY code/baseline/train_DMNC.py /app/
COPY code/baseline/train_Leap.py /app/
COPY code/baseline/train_LR.py /app/
COPY code/baseline/train_Retain.py /app/
# copy data
COPY data/prepare_data/data/ /app/data/
# copy data prep code
COPY data/prepare_data/read_functions.py /app/
COPY data/prepare_data/create_vocabulary.py /app/
COPY data/prepare_data/prepare_data.py /app/
COPY data/prepare_data/construct_adj.py /app/

WORKDIR /app/


RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

#CMD ["python", "train_GAMENet.py", "--model_name", "GAMENet", "--ddi"]
#CMD ["sh", "../gamenet.sh"]
CMD ["sh", "gamenet.sh"]
#CMD ["python", "/app/code/train_GAMENet.py", "--model_name", "GAMENet", "--ddi"]
#CMD ["python", "/app/code/train_GAMENET.py", "--model_name", "GAMENet", "--ddi", "--resume_path", "/app/code/saved/GAMENet/Epoch_39_JA_0.5175_DDI_0.0782.model", "--eval"]

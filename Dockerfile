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


COPY . /app
WORKDIR /app/code/

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

#CMD ["python", "train_GAMENet.py", "--model_name", "GAMENet", "--ddi"]
CMD ["sh", "../train_and_eval_model.sh"]
#CMD ["python", "/app/code/train_GAMENet.py", "--model_name", "GAMENet", "--ddi"]
#CMD ["python", "/app/code/train_GAMENET.py", "--model_name", "GAMENet", "--ddi", "--resume_path", "/app/code/saved/GAMENet/Epoch_39_JA_0.5175_DDI_0.0782.model", "--eval"]

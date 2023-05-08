FROM nvidia/cuda:11.0.3-base-ubuntu20.04

#ARG MIMIC=4
#ENV MIMIC ${MIMIC}

RUN apt-get update
RUN apt-get install -y python3 python3-pip

COPY *.py /app/
COPY requirements.txt /app/

WORKDIR /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#CMD ["python3", "gamenet.py", "--mimic", ${MIMIC}]

#CMD ["python3", "gamenet.py", "--mimic", "$MIMIC"]

ENTRYPOINT ["python3", "gamenet.py"]
#CMD ["--mimic", "4"]

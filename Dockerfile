FROM rapidsai/rapidsai-core:22.08-cuda11.4-base-ubuntu20.04-py3.9
USER root

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get clean \
  && apt-key del 7fa2af80 \
  && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

WORKDIR /app
COPY . /app/

RUN pip3 install -r /app/requirements.txt

FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*
    
RUN ls -l /usr/local/cuda && nvcc --version


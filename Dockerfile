

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Evita prompts do apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalações básicas e dependências do conda
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Instala o Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app
COPY environment.yml /app/environment.yml

RUN pip install jupyterlab

RUN conda env create -f /app/environment.yml

RUN echo "source activate horizon_gs" > ~/.bashrc
ENV PATH /opt/conda/envs/horizon_gs/bin:$PATH

EXPOSE 8888
CMD ["conda", "run", "-n", "horizon_gs", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

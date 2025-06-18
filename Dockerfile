

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt update && apt install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    apt clean
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app

COPY . /app

RUN pip install jupyter

RUN conda env create -f /app/environment.yml

RUN echo "source activate horizon_gs" > ~/.bashrc
ENV PATH /opt/conda/envs/horizon_gs/bin:$PATH

EXPOSE 8888
CMD ["conda", "run", "-n", "horizon_gs", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

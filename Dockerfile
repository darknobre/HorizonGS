

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get update && apt-get install -y git

WORKDIR /app
COPY environment.yml /app/environment.yml

RUN pip install jupyterlab

RUN conda env create -f /app/environment.yml

RUN echo "source activate horizon_gs" > ~/.bashrc
ENV PATH /opt/conda/envs/horizon_gs/bin:$PATH

EXPOSE 8888
CMD ["conda", "run", "-n", "horizon_gs", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]



FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

RUN pip install jupyter

RUN conda env create -f /app/environment.yml

RUN echo "source activate horizon_gs" > ~/.bashrc
ENV PATH /opt/conda/envs/horizon_gs/bin:$PATH

EXPOSE 8888
CMD ["conda", "run", "-n", "horizon_gs", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

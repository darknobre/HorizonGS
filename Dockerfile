
# Use CUDA-compatible base image
FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN pip install jupyter

RUN conda env create -f /app/environment.yml.yml

RUN echo "source activate horizon_gs" > ~/.bashrc
ENV PATH /opt/conda/envs/horizon_gs/bin:$PATH

EXPOSE 8888
CMD ["conda", "run", "-n", "horizon_gs", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

FROM nvidia/cuda:11.2.2-base-ubuntu20.04

ENV MINICONDA_VERSION=py39_4.9.2
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -a

ENV PATH=/opt/conda/bin:$PATH

COPY environment/environment.yml /tmp/environment.yml
COPY environment/requirements.txt /tmp/requirements.txt

RUN conda env create -f /tmp/environment.yml && conda clean -a
RUN /opt/conda/envs/logbert/bin/pip install -r /tmp/requirements.txt

ENV PATH /opt/conda/envs/logbert/bin:$PATH
ENV CONDA_DEFAULT_ENV logbert

COPY . /app
WORKDIR /app/logbert

CMD ["python", "logbert.py", "deploy"]

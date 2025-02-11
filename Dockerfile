# https://github.com/tf63/transformer-study/blob/main/docker/Dockerfile.cu117
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /opt
RUN apt update && \
    apt install -y \
    wget \
    bc \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    sudo

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

COPY environment.yml .
RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -f environment.yml
ENV PATH /opt/miniconda3/envs/task_arithmetic/bin:$PATH

RUN conda run -n task_arithmetic pip install autopep8

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install --user autopep8

ENV PYTHONBREAKPOINT=ipdb.set_trace

ARG USER_UID
ARG USER_GID

ARG USER_NAME=user
ARG GROUP_NAME=user
ARG PASSWD=user

RUN if ! getent group $USER_GID >/dev/null; then \
    groupadd -g $USER_GID $GROUP_NAME; \
    fi

RUN useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USER_NAME

USER $USER_NAME
ENV PATH /home/$USER_NAME/.local/bin:$PATH

ENV HOME /home/$USER_NAME
WORKDIR /app

CMD ["/bin/bash"]

RUN conda init bash && \
    echo "conda activate task_arithmetic" >> ~/.bashrc

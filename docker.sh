#!/bin/bash
# https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/docker.sh

# usage ----------------------------------------------
# bash docker.sh build          # build image
# bash docker.sh shell          # run container as user
# ----------------------------------------------------

PROJECT_NAME="task_arithmetic"
DATASET_DIR="~/dataset"
MODEL_DIR="~/model"

DOCKERFILE_NAME="Dockerfile"


build()
{
    export DOCKER_BUILDKIT=1 
    docker build . -f $DOCKERFILE_NAME --build-arg USER_UID=`(id -u)` --build-arg USER_GID=`(id -g)` -t $PROJECT_NAME
}

shell() 
{
    docker run --gpus all --shm-size=16g -it -v $(pwd):/app -v $DATASET_DIR:/app/dataset -v $MODEL_DIR:/app/model --env-file ./.env $PROJECT_NAME /bin/bash
}

help()
{
    echo "usage: bash docker.sh [build|shell|help]"
}


if [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell
else
    help
fi
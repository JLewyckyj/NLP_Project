# -*- dockerfile -*-
FROM ubuntu:latest


#
# Docker file system locations
#

ARG docker_api_dir=/api
ARG docker_lib_dir=${docker_api_dir}/lib
ARG docker_models_dir=${docker_api_dir}/models
ARG docker_liblinear_dir=${docker_lib_dir}/liblinear-211

RUN mkdir -p ${docker_api_dir} && mkdir -p ${docker_lib_dir} && mkdir -p ${docker_models_dir} && mkdir -p ${docker_liblinear_dir}

#
# Deps
#


# System

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
    build-essential \
    checkinstall \
    curl \
    libbz2-dev \
    libc6-dev \
    libgdbm-dev \
    libncursesw5-dev \
    libreadline-gplv2-dev \
    libsqlite3-dev \
    libssl-dev \
    tk-dev \
    wget \
    git-core \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


# liblinear (patched)

RUN wget https://github.com/cjlin1/liblinear/archive/v211.tar.gz -O ${docker_lib_dir}/liblinear.tgz && tar -zxvf ${docker_lib_dir}/liblinear.tgz -C ${docker_lib_dir}
COPY ./liblinear/001.patch ${docker_liblinear_dir}/
WORKDIR ${docker_liblinear_dir}
RUN ls ${docker_lib_dir}
RUN ls
RUN patch -p1 < 001.patch &&  make clean && make && cd python && make

ENV LIBLINEAR_CLIB="${docker_liblinear_dir}/liblinear.so.3"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./requirements.txt ${docker_api_dir}
RUN pip3 install -r ${docker_api_dir}/requirements.txt


#
# Source code
#


COPY . ${docker_api_dir}
WORKDIR ${docker_api_dir}
COPY ./models/review.dict ${docker_models_dir}
COPY ./models/review.liblinear.model ${docker_models_dir}


ENV MODEL_FILE="${docker_models_dir}/review.liblinear.model"
ENV DICT_FILE="${docker_models_dir}/review.dict"

#
# Entry-point
#

EXPOSE ${HTTP_PORT}
CMD ["sh", "api_entrypoint.sh"]

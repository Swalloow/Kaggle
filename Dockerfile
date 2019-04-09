# Copyright (c) Junyoung Park.

FROM jupyter/tensorflow-notebook:latest

LABEL maintainer="swalloow"

USER root

# Install python-software-properties software-properties-common
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependency gcc/g++
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Install gcc/g++
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc-7 g++-7 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install c++/swig
# RUN apt-get update && \
#     apt-get install -y build-essential swig && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Update alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Conda install xgboost, lightgbm, catboost, gensim
RUN conda install --quiet --yes \
		'conda-build' \
    'lightgbm' \
    'xgboost' \
    'catboost' \
    'gensim' && \
    conda build purge-all && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install python packages
RUN sudo -H $CONDA_DIR/bin/python -m pip install tqdm \
					 spacy \
					 arrow \
					 missingno \
					 requests

# Clean up pip cache
RUN rm -rf /root/.cache/pip/*

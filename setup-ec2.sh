#!/bin/bash
# This script is used to quickly setup conda environment on EC2.
sudo apt-get update -y
sudo apt-get install -y tree zip unzip curl
ssh-keygen -t rsa -b 4096 -N "" -f "/home/ubuntu/.ssh/id_rsa" -C "nguyentp2@fsoft.com.vn" && \
    eval "$(ssh-agent -s)" && \
    ssh-add ~/.ssh/id_rsa
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -u -b -p $HOME/miniconda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh && \
    eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
    conda init && \
    conda config --set auto_activate_base false && \
    source ~/.bashrc
conda create --name ctr python=3.6 -y && \
    source $HOME/miniconda/etc/profile.d/conda.sh && \
    conda activate ctr && \
    conda install -c conda-forge pandas numpy matplotlib scikit-learn lightgbm notebook -y && \
    jupyter notebook --generate-config -y && \
    echo "c.NotebookApp.password = u'sha1:afe3892d748d:a53857c65f24782b7f97abba4568cefb1d476b9f'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/ubuntu'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

FROM tensorflow/tensorflow:latest-devel-gpu

USER root

ENV TF_GPU_ALLOCATOR=cuda_malloc_async

RUN apt update && apt install -y \
python3-pip \
vim \
sudo \
curl \
git 

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


WORKDIR /workspace

COPY Quantum_NTK Quantum_NTK

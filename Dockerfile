FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update && apt-get install -y \
  libsm6 \
  libxrender1 \
  libfontconfig1 \
  libxext6 \
  git \
  vim \
  wget

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade cython
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT bash
# Python version 3.9.18
FROM python@sha256:17d96c91156bd5941ca1b6f70606254f7f98be8dbe662c34f41a0080fd490b0c

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libfftw3-dev \
    liblapack-dev \
    libsndfile-dev \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install cython==0.29.19 \
    && pip install tifresi==0.1.2 \
    && pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./

RUN mkdir -p checkpoints/stylegan2/greatesthits \
    checkpoints/stylegan2/dcase \
    checkpoints/encoder/greatesthits \
    checkpoints/encoder/dcase \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/greatesthits/network-snapshot-002800.pkl -P checkpoints/stylegan2/greatesthits \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/dcase/network-snapshot-002200.pkl -P checkpoints/stylegan2/dcase \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/greatesthits/netE_epoch_best.pth -P checkpoints/encoder/greatesthits \
    && wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/dcase/netE_epoch_best.pth -P checkpoints/encoder/dcase

EXPOSE 8100

WORKDIR /app/interface

COPY ./script.sh /
RUN chmod +x /script.sh
ENTRYPOINT ["/script.sh"]
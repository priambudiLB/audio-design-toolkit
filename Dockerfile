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
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install cython==0.29.19
RUN pip install tifresi==0.1.2

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./
RUN wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/greatesthits/network-snapshot-002800.pkl
RUN mv network-snapshot-002800.pkl checkpoints/stylegan2/greatesthits
RUN wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/audio-stylegan2/dcase/network-snapshot-002200.pkl
RUN mv network-snapshot-002200.pkl checkpoints/stylegan2/dcase
RUN wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/greatesthits/netE_epoch_best.pth
RUN mv netE_epoch_best.pth checkpoints/encoder/greatesthits
RUN wget https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/resources/model_weights/encoder/dcase/netE_epoch_best.pth
RUN mv netE_epoch_best.pth checkpoints/encoder/dcase

EXPOSE 8100

WORKDIR /app/interface

COPY ./script.sh /
RUN chmod +x /script.sh
ENTRYPOINT ["/script.sh"]
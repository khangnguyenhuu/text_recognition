FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

COPY . /ocr/
WORKDIR /ocr/
RUN apt-get update \
    && apt-get install python3.8 -y \
    && apt-get update \
    && apt-get install python3-pip -y \
    && python3 -m pip install --upgrade pip \
    && apt-get update \
    && apt install -y libgl1-mesa-glx \
    && pip3 install -r requirements.txt \
    && pip3 install gdown \
    && export LC_ALL="en_US.UTF-8" && export LC_CTYPE="en_US.UTF-8" \
    && apt-get install zip -y \
    && cd ./src && gdown --id 1C9LBHkhu1IzQJHaq0vz5SdqKkC_vAD56 \
    && unzip ./models && cd ..

CMD python3 predict.py

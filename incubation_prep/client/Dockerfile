FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1 ffmpeg

ARG JINA_VERSION=

RUN python3 -m pip install --no-cache-dir jina${JINA_VERSION:+==${JINA_VERSION}}

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt
# Install YOLOv5-pip
RUN git clone https://github.com/Tien-Cheng/bytetrack-pip.git
RUN pip install bytetrack-pip/

RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/



ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV KAFKA_METRICS_TOPIC="metrics"

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["sh"]
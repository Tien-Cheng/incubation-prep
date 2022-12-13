FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

ARG JINA_VERSION=

RUN python3 -m pip install --no-cache-dir jina${JINA_VERSION:+==${JINA_VERSION}}

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt
# Install YOLOv5-pip
RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/

ENV EXECUTOR_NAME="detector"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV KAFKA_METRICS_TOPIC="metrics"
ENV REDIS_HOST="localhost"
ENV REDIS_PORT="6379"
ENV REDIS_DB="0"

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
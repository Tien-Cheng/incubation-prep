FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/

RUN git clone https://github.com/Tien-Cheng/bytetrack-pip.git
RUN pip install bytetrack-pip/

COPY . /workdir/
WORKDIR /workdir

ENV EXECUTOR_NAME="tracker"
ENV BROKER="kafka"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV ZMQ_HOSTNAME="*"
ENV ZMQ_PORT_IN="5002"
ENV ZMQ_PORT_OUT="5003"
ENV KAFKA_CONSUMER_GROUP="track"
ENV KAFKA_CONSUME_TOPIC="dets"
ENV KAFKA_PRODUCE_TOPIC="tracks"
ENV KAFKA_METRICS_TOPIC="metrics"
ENV REDIS_HOST="localhost"
ENV REDIS_PORT="6379"
ENV REDIS_DB="0"

EXPOSE 5003
EXPOSE 5002

ENTRYPOINT ["python", "track.py"]
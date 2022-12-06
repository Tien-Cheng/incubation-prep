FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

# Install YOLOv5-pip
COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/



COPY . /workdir/
WORKDIR /workdir

ENV EXECUTOR_NAME="detector"
ENV BROKER="kafka"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV ZMQ_HOSTNAME="*"
ENV ZMQ_PORT_IN="5001"
ENV ZMQ_PORT_OUT="5002"
ENV KAFKA_CONSUMER_GROUP="dets"
ENV KAFKA_CONSUME_TOPIC="frames"
ENV KAFKA_PRODUCE_TOPIC="dets"
ENV KAFKA_METRICS_TOPIC="metrics"

EXPOSE 5001
EXPOSE 5002
ENTRYPOINT ["python", "det.py"]
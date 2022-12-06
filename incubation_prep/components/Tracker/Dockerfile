FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENV BROKER="kafka"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV ZMQ_HOSTNAME="*"
ENV ZMQ_PORT_IN="5002"
ENV ZMQ_PORT_OUT="5003"
ENV KAFKA_CONSUMER_GROUP="track"
ENV KAFKA_CONSUME_TOPIC="det_out"
ENV KAFKA_PRODUCE_TOPIC="track_out"

EXPOSE 5003
EXPOSE 5002

ENTRYPOINT ["python", "track.py"]
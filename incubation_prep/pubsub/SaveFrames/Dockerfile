FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git ffmpeg libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENV BROKER="kafka"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV ZMQ_HOSTNAME="*"
ENV ZMQ_PORT_IN="5003"
ENV ZMQ_PORT_OUT="5004"
ENV KAFKA_CONSUMER_GROUP="save"
ENV KAFKA_CONSUME_TOPIC="tracks"

EXPOSE 5555
EXPOSE 5003
EXPOSE 5004

ENTRYPOINT ["python", "save.py"]
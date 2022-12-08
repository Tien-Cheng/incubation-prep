FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git ffmpeg libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/

COPY . /workdir/
WORKDIR /workdir

ENV EXECUTOR_NAME="monolith"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV KAFKA_METRICS_TOPIC="metrics"

EXPOSE 8554
EXPOSE 5555
EXPOSE 5003
EXPOSE 5004
ENTRYPOINT ["python"]
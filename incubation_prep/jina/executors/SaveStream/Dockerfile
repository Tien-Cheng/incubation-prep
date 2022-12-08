ARG JINA_VERSION=latest

FROM jinaai/jina:${JINA_VERSION}

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git ffmpeg libgl1

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

ENV EXECUTOR_NAME="save_stream"
ENV KAFKA_ADDRESS="127.0.0.1:9092"
ENV KAFKA_METRICS_TOPIC="metrics"

# setup the workspace
COPY . /workdir/
WORKDIR /workdir

EXPOSE 8554
EXPOSE 5555
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
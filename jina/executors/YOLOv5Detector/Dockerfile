ARG JINA_VERSION=latest

FROM jinaai/jina:${JINA_VERSION}

# Install ScaledYOLOv4
RUN git clone https://github.com/Tien-Cheng/ScaledYOLOv4.git
RUN pip install ScaledYOLOv4/

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

# setup the workspace
COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
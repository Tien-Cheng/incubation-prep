FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

# Install YOLOv5-pip
COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

RUN git clone https://github.com/Tien-Cheng/yolov5-pip.git
RUN pip install yolov5-pip/



COPY . /workdir/
WORKDIR /workdir

EXPOSE 4001
ENTRYPOINT ["python", "-m", "uvicorn", "detector:app", "--host=0.0.0.0", "--port=4001"]
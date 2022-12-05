FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git ffmpeg libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

EXPOSE 8554
EXPOSE 5555
EXPOSE 4003
ENTRYPOINT ["python", "-m", "uvicorn", "output:app", "--host=0.0.0.0", "--port=4003"]
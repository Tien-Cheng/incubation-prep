FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git libgl1

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

EXPOSE 4002
ENTRYPOINT ["python", "-m", "uvicorn", "track:app", "--host=0.0.0.0", "--port=4002"]
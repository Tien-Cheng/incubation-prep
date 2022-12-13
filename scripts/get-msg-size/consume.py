from confluent_kafka import Consumer
from docarray import DocumentArray
import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

consumer = Consumer({
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": "frames",
    "client.id": "consumer_1",
    "auto.offset.reset": "smallest",
})
consumer.subscribe(["frames"])
try:
    with open("consumer_1.log", "w") as f:
        while True:
            data = consumer.poll(timeout=1)
            if data is None:
                continue
            if data.error():
                print(data.error())
            else:
                data = data.value()
                print(f"Consumer 1 Received data with size {convert_size(len(data))}")
                docarray = DocumentArray.from_bytes(data)
                print(docarray.summary())
except:
    pass
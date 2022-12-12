from time import sleep
from confluent_kafka import Producer

producer = Producer({"bootstrap.servers": "127.0.0.1:9092"})

n = 100
topic = "number_producer"
producer.purge()
for i in range(n):
    producer.produce(topic, value=str(i).encode("utf-8"), on_delivery=lambda x, y : print(f"Error: {x}, Msg: {y.value()}"))
    producer.poll(0)
    sleep(2)

from confluent_kafka import Consumer

consumer = Consumer(
    {
        "bootstrap.servers": "127.0.0.1:9092",
        "group.id": "numbers",
        "client.id": "consumer_1",
        "auto.offset.reset": "smallest",
    }
)
consumer.subscribe(["number_producer"])

try:
    with open("consumer_1.log", "w") as f:
        while True:
            data = consumer.poll(timeout=1)
            if data is None:
                continue
            if data.error():
                print(data.error())
            else:
                data = str(data.value(), "utf-8")
                print(f"Consumer 1 Received {data}")
                f.write(data + "\n")
except:
    pass

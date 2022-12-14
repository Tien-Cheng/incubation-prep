from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from time import sleep

client = KafkaAdminClient()
topics = [
    "dets",
    "frames",
    "tracks",
    "out"
]
try:
    client.delete_topics(topics)
except:
    pass
sleep(5)
client.create_topics(
    [NewTopic(name, 3, 1) for name in topics]
)
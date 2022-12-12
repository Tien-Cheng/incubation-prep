from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from time import sleep

client = KafkaAdminClient()
try:
    client.delete_topics(["number_producer"])
except:
    pass
sleep(5)
client.create_topics(
    [NewTopic("number_producer", 2, 1)]
)
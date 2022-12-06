from kafka.admin import KafkaAdminClient, NewTopic

client = KafkaAdminClient(
    bootstrap_servers="127.0.0.1:9092"
)
print(client.list_topics())

client.create_topics(
    [NewTopic("metrics", 3, 1), NewTopic("tracks", 3, 1), NewTopic("dets", 3, 1), NewTopic("frames", 3, 1)]
)

from confluent_kafka.admin import AdminClient

client = AdminClient(
   { "bootstrap.servers":"127.0.0.1:9092" }
)

client.create_topics(
    ["dets", "frames", "tracks"]
)

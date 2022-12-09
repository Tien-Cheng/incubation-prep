build:
	cp ../component.py . && cp ../zmq_subscriber.py . && docker build . -t pubsub-stream-output:1.1
run:
	docker run --gpus all  -it --rm --net=host pubsub-stream-output:1.1
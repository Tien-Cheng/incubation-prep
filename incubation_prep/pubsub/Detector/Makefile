build:
	cp ../component.py . && cp ../zmq_subscriber.py . && docker build . -t pubsub-detector:1.3
run:
	docker run --gpus all -it --rm --net=host pubsub-detector:1.3
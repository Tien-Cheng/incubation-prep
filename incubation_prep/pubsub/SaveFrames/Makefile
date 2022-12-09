build:
	cp ../component.py . && cp ../zmq_subscriber.py . && docker build . -t pubsub-save-frames:1.1
run:
	docker run --gpus all -it --rm --net=host pubsub-save-frames:1.1
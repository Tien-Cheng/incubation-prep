build:
	cp ../component.py . && cp ../zmq_subscriber.py . && docker build . -t pubsub-tracker:1.2
run:
	docker run --gpus all -it --rm --net=host pubsub-tracker:1.2
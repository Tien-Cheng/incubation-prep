from jina import DocumentArray, Executor, requests


class ObjectDetector(Executor):
    """Takes in a frame /image and returns detections"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
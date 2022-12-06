import zmq
import zmq.asyncio
import imagezmq
import threading

class DocumentSubscriber:
    def __init__(self, hostname, port):
        raise NotImplementedError
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    async def _run(self):
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.hostname}:{self.port}")
        socket.setsockopt(zmq.SUBSCRIBE, b'')

        while not self._stop:
            # self._data = receiver.recv_image()
            self._data = await socket.recv() # receive bytes
            self._data_ready.set()
        socket.close()

    def close(self):
        self._stop = True

# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            # self._data = receiver.recv_image()
            self._data = receiver.zmq_socket.recv() # receive bytes
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True

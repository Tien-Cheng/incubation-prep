from threading import Thread
import cv2, time
import imagezmq

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = imagezmq.ImageHub(src)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            (self.status, self.frame) = self.capture.recv_image()
            self.capture.send_reply()
            time.sleep(self.FPS)
            
    def show_frame(self):
        cv2.imshow('frame', cv2.resize(self.frame, (1280, 720)))
        cv2.waitKey(self.FPS_MS)

if __name__ == '__main__':
    src = 'tcp://*:5555'
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass
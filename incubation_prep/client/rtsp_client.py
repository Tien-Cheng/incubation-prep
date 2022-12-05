import datetime
import time

import cv2
from vidgear.gears import CamGear, VideoGear


class Reconnecting_VideoGear:
    def __init__(self, cam_address, reset_attempts=50, reset_delay=5):
        self.cam_address = cam_address
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.source = CamGear(source=self.cam_address).start()
        self.running = True

    def read(self):
        if self.source is None:
            return None
        if self.running and self.reset_attempts > 0:
            frame = self.source.read()
            if frame is None:
                self.source.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.source = CamGear(source=self.cam_address).start()
                # return previous frame
                return self.frame
            else:
                self.frame = frame
                return frame
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.source is None:
            self.source.stop()


if __name__ == "__main__":
    # open any valid video stream
    stream = Reconnecting_VideoGear(
        cam_address="rtsp://localhost:8554/test2",
        reset_attempts=20,
        reset_delay=5,
    )

    cv2.namedWindow("output_rtsp", cv2.WINDOW_NORMAL)
    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if None-type
        if frame is None:
            break

        # {do something with the frame here}
        cv2.resize(frame, (1280, 720))

        # Show output window
        cv2.imshow("output_rtsp", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

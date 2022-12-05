import cv2
from vidgear.gears import NetGear

# define Netgear Client with `receive_mode = True` and default parameter
client = NetGear(receive_mode=True, address="127.0.0.1", port=5555)

cv2.namedWindow("output_zmq", cv2.WINDOW_NORMAL)
# loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    cv2.resize(frame, (1280, 720))
    # Show output window
    cv2.imshow("output_zmq", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()

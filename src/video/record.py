import cv2
import sys

# Read video
video = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mjpg', fourcc, 20.0, (640, 480))

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

template = cv2.imread('ball.png', 0)

while (True):
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # write the flipped frame
    out.write(frame)

    cv2.imshow('Frame', frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break


video.release()
out.release()
cv2.destroyAllWindows()

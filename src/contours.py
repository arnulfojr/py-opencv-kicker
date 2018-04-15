# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import sys
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = np.array([145, 150, 200])
greenUpper = np.array([165, 255, 255])
pts = deque(maxlen=args["buffer"])

# grab the current frame
# Read video
video = cv2.VideoCapture('game.mov')

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()

# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize(frame, height=600)
# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow("Frame", mask)
key = cv2.waitKey(0) & 0xFF
# find contours in the mask and initialize the current
# (x, y) center of the ball

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]
center = None

pp = []
# only proceed if at least one contour was found
if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if 8 < radius < 15:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            pp.append((int(x), int(y)))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

# update the points queue
pts.appendleft(center)
# loop over the set of tracked points
for i in range(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
    if pts[i - 1] is None or pts[i] is None:
        continue

    # otherwise, compute the thickness of the line and
    # draw the connecting lines
    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

# show the frame to our screen
cv2.imshow("Frame", frame)
key = cv2.waitKey(0) & 0xFF

cent = (sum([p[0] for p in pp]) / len(pp), sum([p[1] for p in pp]) / len(pp))
# sort by polar angle
pp.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

cv2.line(frame, pp[0], pp[1], (255, 0, 0), 5)
cv2.line(frame, pp[1], pp[2], (255, 0, 0), 5)
cv2.line(frame, pp[2], pp[3], (255, 0, 0), 5)
cv2.line(frame, pp[3], pp[0], (255, 0, 0), 5)

cv2.imshow("Frame", frame)
key = cv2.waitKey(0) & 0xFF

# cleanup the camera and close any open windows
cv2.destroyAllWindows()

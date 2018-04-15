import cv2
import numpy as np
import sys
import math

# Read video
video = cv2.VideoCapture('game.mov')

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

counter = 0
BUFFER_SIZE = 64

LOWER_HUE = 140
UPPER_HUE = 170
LOWER_COLOR = np.array([LOWER_HUE, 150, 200])
UPPER_COLOR = np.array([UPPER_HUE, 255, 255])

CALIBRATION_BOX_MIN_RADIUS = 8
CALIBRATION_BOX_MAX_RADIUS = 20

BALL_MIN_RADIUS = 20
BALL_MAX_RADIUS = 26


def detect_box(output):
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
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

            # only proceed if the radius meets a minimum size
            if CALIBRATION_BOX_MIN_RADIUS < radius < CALIBRATION_BOX_MAX_RADIUS:
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                pp.append((int(x), int(y)))
                cv2.circle(output, center, 5, (0, 0, 255), -1)

    if len(pp) > 3:
        cent = (sum([p[0] for p in pp]) / len(pp),
                sum([p[1] for p in pp]) / len(pp))
        # sort by polar angle
        pp.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

        cv2.line(output, pp[0], pp[1], (255, 0, 0), 5)
        cv2.line(output, pp[1], pp[2], (255, 0, 0), 5)
        cv2.line(output, pp[2], pp[3], (255, 0, 0), 5)
        cv2.line(output, pp[3], pp[0], (255, 0, 0), 5)
    else:
        # Display error message
        cv2.putText(output, "No contour detected", (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


def add_circles(frame, detected_circles, output):
    # ensure at least some circles were found
    if detected_circles is None:
        return frame, None

    # convert the (x, y) coordinates and radius of the circles to integers
    detected_circles = np.round(detected_circles[0, :]).astype("int")

    def draw_circle_in_image(x, y, r):
        print(x, y, r)
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output,
                      (x - 5, y - 5),
                      (x + 5, y + 5),
                      (0, 128, 255), -1)

    # loop over the (x, y) coordinates and radius of the circles
    [draw_circle_in_image(x, y, r) for (x, y, r) in detected_circles
     if r > 0]

    return frame, circles


while True:
    # Read first frame.
    ok, output = video.read()
    counter = counter + 1
    if not ok:
        print('Cannot read video file')
        sys.exit()
    # start timer
    timer = cv2.getTickCount()

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 75, 75)

    # detect circles in the image
    # 24-30 min-max
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=20, minRadius=BALL_MIN_RADIUS,
                               maxRadius=BALL_MAX_RADIUS)

    # detect the outer box based on color points
    detect_box(output)

    # add the circles to the frame
    add_circles(blur, circles, output)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(output, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    # show the output image
    cv2.imshow("output", output)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()

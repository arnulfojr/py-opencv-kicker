import cv2
import numpy as np
import sys

# Read video
video = cv2.VideoCapture('game.mov')

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

counter = 0

while True:
    # Read first frame.
    ok, image = video.read()
    counter = counter + 1
    if not ok:
        print('Cannot read video file')
        sys.exit()

    output = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 75, 75)

    # detect circles in the image
    # 24-30 min-max
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=20, minRadius=20, maxRadius=25)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            if r > 0:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(blur, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(blur, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image
    cv2.imshow("output", blur)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()

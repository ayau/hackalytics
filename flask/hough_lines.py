# Hough line transform to detect track lines
# Based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np
import time

import cv2
import numpy as np

img = cv2.imread('../sample_videos/screen_1_2160x3840px.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,100,apertureSize = 3)

# These parameters are adjusted to detect only the longest lines (ex: track lines)
minLineLength = 60
maxLineGap = 15
lines = cv2.HoughLinesP(edges,1,np.pi/1800,160,minLineLength,maxLineGap)


# print(lines)

for line in lines:
    # print(line)
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0, 255),2)

# Calculate the angles of all the lines

# Throw out outliers

# Calculate the distances between lines

# Throw out outliers

# Sort distances

# Perform plane fitting

cv2.imshow('lines',  cv2.resize(img, (1920, 1080)))

# Exit OpenCV
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

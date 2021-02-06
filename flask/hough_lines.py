# Hough line transform to detect track lines
# Based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np
import time

import cv2
import numpy as np

img = cv2.imread('../sample_videos/screen_6_2160x3840px.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,100,apertureSize = 3)

# These parameters are adjusted to detect only the longest lines (ex: track lines)
minLineLength = 800
maxLineGap = 270
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)


# print(lines)

for line in lines:
    # print(line)
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0, 255),2)

cv2.imshow('lines',  cv2.resize(img, (1920, 1080)))

while True:
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

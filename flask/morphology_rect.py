import cv2
import numpy as np

# read image
img = cv2.imread("../sample_videos/screen_1_2160x3840px.jpg")

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255-gray

# blur image
blur = cv2.GaussianBlur(gray, (3,3), 0)

# do adaptive threshold on gray image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
thresh = 255-thresh

# apply morphology
kernel = np.ones((5,5), np.uint8)
rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

# thin
kernel = np.ones((5,5), np.uint8)
rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

# get largest contour
contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for c in contours:
    area_thresh = 0
    area = cv2.contourArea(c)
    if area > area_thresh:
        area = area_thresh
        big_contour = c

# get rotated rectangle from contour
rot_rect = cv2.minAreaRect(big_contour)
box = cv2.boxPoints(rot_rect)
box = np.int0(box)
for p in box:
    pt = (p[0],p[1])
    print(pt)

# draw rotated rectangle on copy of img
rot_bbox = img.copy()
cv2.drawContours(rot_bbox,[box],0,(0,0,255),2)

# write img with red rotated bounding box to disk

cv2.imshow('THRESHOLD',  cv2.resize(thresh, (1920, 1080)))
cv2.imshow('RECT',  cv2.resize(rect, (1920, 1080)))
cv2.imshow('BBOX',  cv2.resize(rot_bbox, (1920, 1080)))

# cv2.imwrite("rectangle_thresh.png", thresh)
# cv2.imwrite("rectangle_outline.png", rect)
# cv2.imwrite("rectangle_bounds.png", rot_bbox)

# display it
# cv2.imshow("IMAGE", img)
# cv2.imshow("THRESHOLD", thresh)
# cv2.imshow("RECT", rect)
# cv2.imshow("BBOX", rot_bbox)


cv2.waitKey(0)

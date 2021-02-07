# Hough line transform to detect track lines
# Based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np

img = cv2.imread('../sample_videos/screen_1_2160x3840px.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove the top half of the image
mask = np.zeros(gray.shape[:2], np.uint8)
start_rect = (0, int(img.shape[0]/2))
end_rect = (int(img.shape[1]), int(img.shape[0]))
mask = cv2.rectangle(mask, start_rect, end_rect, (255, 255, 255), -1)
gray = cv2.bitwise_and(gray, gray, mask=mask)

# Add some noise to remove fine details / lines
smooth = cv2.GaussianBlur(gray, (11, 11), 0) # must be an odd number

# Extract white colors
_, white = cv2.threshold(smooth, 180, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) # elliptical shaped kernel
edges = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

# Hardcoded for now but we should extract it from bg_subtract script
jump_coords = (782 * 4, 384 * 4)
land_coords = (352 * 4, 388 * 4)


# Find lines around the jumping / landing coords
paddingx = int(300 * img.shape[1] / 3840)
paddingy = int(100 * img.shape[0] / 2160)
mask = np.zeros(gray.shape[:2], np.uint8)
start_rect = (int(jump_coords[0] - paddingx), int(jump_coords[1] - paddingy))
end_rect = (int(jump_coords[0] + paddingx), int(jump_coords[1] + paddingy))
mask = cv2.rectangle(mask, start_rect, end_rect, (255, 255, 255), -1)
start_rect = (int(land_coords[0] - paddingx), int(land_coords[1] - paddingy))
end_rect = (int(land_coords[0] + paddingx), int(land_coords[1] + paddingy))
mask = cv2.rectangle(mask, start_rect, end_rect, (255, 255, 255), -1)
edges = cv2.bitwise_and(edges, edges, mask=mask)


# Only identiy the two largest shapes (two lines)
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(cnts) > 0:
    cnt = max(cnts, key=cv2.contourArea)
    output = np.zeros(edges.shape, np.uint8)
    cv2.drawContours(output, [cnt], -1, 255, cv2.FILLED)
    cnts.remove(cnt)
    cnt = max(cnts, key=cv2.contourArea)
    cv2.drawContours(output, [cnt], -1, 255, cv2.FILLED)
    edges = cv2.bitwise_and(edges, output)


backtorgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# Draw take off / landing
cv2.circle(backtorgb, jump_coords, 5, (0, 255, 0), 5)
cv2.circle(backtorgb, land_coords, 5, (0, 255, 0), 5)

cv2.imshow('gra', backtorgb)
# cv2.imshow('lines',  cv2.resize(img, (1920, 1080)))

# Exit OpenCV
while True:
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

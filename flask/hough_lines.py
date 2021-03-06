# Hough line transform to detect track lines
# Based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np
import math

def display_lines_and_step(img, jump_coords, land_coords):
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

    # Find lines around the jumping / landing coords
    paddingx = int(400 * img.shape[1] / 3840)
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
    if len(cnts) > 1:
        cntsSorted = sorted(cnts, key=cv2.contourArea)
        output = np.zeros(edges.shape, np.uint8)
        cv2.drawContours(output, [cntsSorted[-1]], -1, 255, cv2.FILLED)
        cv2.drawContours(output, [cntsSorted[-2]], -1, 255, cv2.FILLED)
        edges = cv2.bitwise_and(edges, output)

    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 100, minLineLength, maxLineGap)

    backtorgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Detect jump distance
    slope_tol = 0.2
    point_tol = 20
    unique_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = math.atan((y2 - y1) / (x2 - x1))
            mid = (y2 - y1) / 2 + y1
            uniq = True
            for uline in unique_lines:
                ux1, uy1, ux2, uy2 = uline[0]
                uslope = math.atan((uy2 - uy1) / (ux2 - ux1))
                umid = (uy2 - uy1) / (ux2 - ux1) * ((x2 - x1)/2 + x1 - ux1) + uy1
                if abs(uslope - slope) < slope_tol and abs(mid - umid) < point_tol:
                    uniq = False
                    break

            if uniq:
                unique_lines.append(line)
                cv2.line(backtorgb, (x1, y1), (x2, y2), (255, 0, 0, 255), 2)
        print(unique_lines)
        if len(unique_lines) == 2:
            intersects = []
            # Find intersection between jump/land line and lane lines
            for line in unique_lines:
                x1, y1, x2, y2 = line[0]
                p = seg_intersect(np.array([x1, y1]), np.array([x2, y2]), np.array([jump_coords[0], jump_coords[1]]), np.array([land_coords[0], land_coords[1]]))
                intersects.append(p)

            scale = 105.5 / dist(intersects[0], intersects[1])
            jump_dist = dist(jump_coords, land_coords) * scale
            left_to_right = jump_coords[0] < land_coords[0]

            if dist(intersects[0], jump_coords) < dist(intersects[1], jump_coords):
                jump_margin = dist(intersects[0], jump_coords) * scale * get_sign(jump_coords, intersects[0], left_to_right)
                land_margin = dist(intersects[1], land_coords) * scale * get_sign(land_coords, intersects[1], left_to_right)
            else:
                jump_margin = dist(intersects[1], jump_coords) * scale * get_sign(jump_coords, intersects[1], left_to_right)
                land_margin = dist(intersects[0], land_coords) * scale * get_sign(land_coords, intersects[0], left_to_right)

            print('Jump Distance: ', jump_dist)
            print('Jump Margin (positive means jumped early): ', jump_margin)
            print('Land Margin (positive means landed early): ', land_margin)


    # Draw take off / landing
    cv2.circle(backtorgb, jump_coords, 5, (0, 255, 0), 5)
    cv2.circle(backtorgb, land_coords, 5, (0, 255, 0), 5)

    return backtorgb
    # cv2.imshow('lines',  cv2.resize(img, (1920, 1080)))

def get_sign(jump, line, ltr):
    if ltr:
        return 1 if jump[0] < line[0] else -1
    else:
        return 1 if jump[0] > line[0] else -1

def dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

data = [
    # jump coords, land coords
    [(782 * 4, 384 * 4), (352 * 4, 388 * 4)],
    [(827 * 4, 359 * 4), (383 * 4, 366 * 4)],
    [(908 * 4, 363 * 4), (405 * 4, 369 * 4)],
    [(0,0), (0,0)], #nothing returned for video 4
    [(290 * 4, 372 * 4), (75 * 4, 368 * 4)], 
    [(284 * 4, 373 * 4), (81 * 4, 374 * 4)]

]

for i in range(6):
    if i == 3: continue # 4th video is fucked
    folder = '../sample_videos/'
    video_name = folder + 'jump' + str(i+1) + '.mp4' # if we use small videos, we need to adjust the smooth = cv2.GaussianBlur(gray, (11, 11), 0) to be smaller, like 3
    cap = cv2.VideoCapture(video_name)
    _, img = cap.read()
    frame_path = "../media/"+str(i+1)+"/stats/ground_markings.jpg"
    cv2.imwrite(frame_path, display_lines_and_step(img, data[i][0], data[i][1]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # cv2.imshow(str(i), display_lines_and_step(img, data[i][0], data[i][1]))

while True:
    # Exit OpenCV
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

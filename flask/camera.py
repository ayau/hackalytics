import cv2
import math
import numpy as np

from scipy.spatial.transform import Rotation

# From GL:
# gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);


def projectionMatrix(fov, aspect, near, far):
    yScale = math.tan(fov/2 * (math.pi / 180))
    xScale = yScale / aspect
    nearmfar = near - far

    return np.array([
        [xScale, 0, (far + near) / nearmfar, -1],
        [0, yScale, 2*far*near / nearmfar, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype="double")

# M = [X/2,   0, 0, X/2,
#    0, Y/2, 0, Y/2,
#    0,   0, 1,   0,
#    0,   0, 0,   1]
def viewportMatrix(width, height):
    return np.array([
        [width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype="double")

### CAMERA ROTATION
r = Rotation.from_euler('xyz', [80.6, -0.266, 87.2], degrees=True)

#r = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
rm = r.as_matrix()

### CAMERA POSITION
t = np.array([63.5,0.85772,42.629])

V = np.empty((4, 4))
V[:3, :3] = rm
V[:3, 3] = t
V[3, :] = [0, 0, 0, 1]


im = cv2.imread('../sample_videos/screen_1_2160x3840px.jpg')
size = im.shape
center = (size[1] / 2, size[0] / 2)
fov = 93.4
focal_length = (size[1] / 2) / math.tan(fov/2 * math.pi / 180)

viewport_matrix = viewportMatrix(size[0], size[1])

# print(size)
origin_point = (1809, 1903)


px = center[1]
py = center[0]

projection_matrix = projectionMatrix(93.4, size[0] / size[1], 0.0001, 1000)

# np.array([
#     [focal_length, 0, px, 0],
#     [0, focal_length, py, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1],
# ], dtype="double")
print(projection_matrix)

# From Blender

# Camera position:


view_proj_matrix = np.matmul(projection_matrix, np.linalg.inv(V))


def point_2d_to_3d(point):
    p = np.array([[point[0], point[1], 1]]).T
    return np.matmul(camera_matrix.T, p)
def point_3d_to_2d(point):
    print(view_proj_matrix)
    p = np.array([[point[0], point[1], point[2], 1]]).T
    transformed = np.matmul(view_proj_matrix, p)
    z_div = transformed[3][0]
    transformed = np.array([transformed[0][0] / z_div, transformed[1][0] / z_div, transformed[2][0] / z_div, 1.0]).T
    return np.matmul(viewport_matrix, transformed)

    

print('')
print(point_3d_to_2d([0, 0, 0]))

print(origin_point)

#  ( 0.0486,  0.9988,  0.0046,  -1.2621)
            # (-0.1641,  0.0034,  0.9864,  -9.6424)
            # ( 0.9853, -0.0487,  0.1640, -21.1881)
            # (-0.0000,  0.0000, -0.0000,   1.0000)

import cv2
import math
import numpy as np

from scipy.spatial.transform import Rotation

# From GL:
# gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);

#
# Camera position
#

### CAMERA ROTATION
r = Rotation.from_euler('xyz', [80.6, -0.266, 87.2], degrees=True)
rm = r.as_matrix()

### CAMERA POSITION
t = np.array([63.5,0.85772,42.629])

V = np.empty((4, 4))
V[:3, :3] = rm
V[:3, 3] = t
V[3, :] = [0, 0, 0, 1]

# print(V)


im = cv2.imread('../sample_videos/screen_1_2160x3840px.jpg')
size = im.shape
center = (size[1] / 2, size[0] / 2)
fov = 93.4
focal_length = (size[1] / 2) / math.tan(fov/2 * math.pi / 180)



# print(size)
origin_point = (1809, 1903)


px = center[1]
py = center[0]

projection_matrix = np.array([
    [focal_length, 0, px, 0],
    [0, focal_length, py, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype="double")
print(projection_matrix)

# From Blender

# Camera position:


view_proj_matrix = np.matmul(projection_matrix, V)


def point_2d_to_3d(point):
    p = np.array([[point[0], point[1], 1]]).T
    return np.matmul(camera_matrix.T, p)
def point_3d_to_2d(point):
    print(view_proj_matrix)
    p = np.array([[point[0], point[1], point[2], 1]]).T
    print(V)
    return np.matmul(view_proj_matrix, p)

print('')
print(point_3d_to_2d([0, 0, 0]))

print(origin_point)

#  ( 0.0486,  0.9988,  0.0046,  -1.2621)
            # (-0.1641,  0.0034,  0.9864,  -9.6424)
            # ( 0.9853, -0.0487,  0.1640, -21.1881)
            # (-0.0000,  0.0000, -0.0000,   1.0000)

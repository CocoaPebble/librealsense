#####################################################
##       RGB-D camera measures back surface        ##
#####################################################

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import png
import pyrealsense2 as rs
from cv2 import aruco


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def is_tangent_line(y1, k, b, xrange):
    if len(y1) != len(xrange):
        print('x and y range are not the same')
    for i, x in enumerate(xrange):
        if y1[i] > (k*x + b) + 0.02:
            return False
    return True


def get_BSR_fit(pts_groups, level, draw=False):
    arr = []
    for pts in pts_groups[level]:
        arr.append(pts)
    np_arr = np.array(arr)
    np_arr = np_arr[np_arr[:, 0].argsort()]

    x = np_arr[:, 0]
    z = np_arr[:, 1]
    weights = np.polyfit(x, z, 16)
    f_poly = np.poly1d(weights)
    f_ploy_der = f_poly.deriv(1)

    sample_num = 300
    xp = np.linspace(x.min(), x.max(), sample_num)
    yp = f_poly(xp)
    yp_der = f_ploy_der(xp)
    boundary_index = np.argwhere(xp > 0)[0][0]

    for i in range(boundary_index):
        for j in range(boundary_index, sample_num):
            if abs(yp_der[i] - yp_der[j]) < 0.01:
                x1 = xp[i]
                x2 = xp[j]
                y1 = yp[i]
                y2 = yp[j]
                k = (y2 - y1) / (x2 - x1)
                b = y2 - k * x2
                der = (yp_der[i] + yp_der[j]) / 2

                if abs(k - der) < 0.02 and is_tangent_line(yp, k, b, xp):
                    theta = np.degrees(np.arctan(k))
                    print(level, 'theta =', theta)

                    if draw:
                        fig, ax = plt.subplots()
                        ax.scatter(x, z, facecolor='None',
                                   edgecolor='k', alpha=0.3)
                        ax.plot(xp, yp)
                        plt.title(f'Cross Section n={level}')
                        plt.plot(x1, y1, 'ro')
                        plt.plot(x2, y2, 'bo')
                        plt.axline([x1, y1], [x2, y2])
                        plt.show()

                    return theta

    print(level, 'yp_der =', yp_der[boundary_index])
    return yp_der[boundary_index]


def check_valid_BSR(xp, yp, x_range, boundary_index):
    ...


def get_points_group(pts, layers, torso_length):
    pts_by_level = [[] for i in range(layers)]
    alpha = torso_length / layers

    for p in pts:
        n = int(p[1] // alpha)
        if n >= 0 and n < layers:
            pts_by_level[n].append([p[0], p[2]])

    return pts_by_level


def integrate_homo_matrix(nx, ny, nz, t):
    T = np.zeros((4, 4))
    T[0, 0] = nx[0]
    T[1, 0] = nx[1]
    T[2, 0] = nx[2]
    T[0, 1] = ny[0]
    T[1, 1] = ny[1]
    T[2, 1] = ny[2]
    T[0, 2] = nz[0]
    T[1, 2] = nz[1]
    T[2, 2] = nz[2]

    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    T[3, 3] = 1

    return T


def get_tp(l1, l2, l3, l4):
    # T-PSCS to D
    x = np.subtract(l4, l3)
    y = np.subtract(l1, l2)
    nx = x / np.linalg.norm(x)
    ny = y / np.linalg.norm(y)
    nz = np.cross(nx, ny)
    print(f'angle between x-axis and y-axis is {angle_between(nx, ny)}, xz is {angle_between(nx, nz)}, yz is {angle_between(ny, nz)}')
    TP = integrate_homo_matrix(nx, ny, nz, l2)
    return TP


def get_inv_tp(tp):
    # D to T-PSCS
    return np.linalg.inv(tp)


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def imageToPointCloud(color_image, depth_image):
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    return pcd



# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 2  # meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
curr_frame = 0

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        pcd = imageToPointCloud(color_image, depth_image)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
        #               [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        # aruco marker detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(
            color_image.copy(), corners, ids)

        if not corners or ids.size != 4:
            print(ids.size)
            print('waiting for 4 landmarks being detected...')
            continue

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        # depth image is 1 channel, color is 3 channels
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (
            depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((frame_markers, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key == ord('i'):
            print(f'depth image shape is {depth_image.shape}')
            print(f'color image shape is {color_image.shape}')
            print(f'depth_intrin is {depth_intrin}')
            print(f'color_intrin is {color_intrin}')

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        if key == ord('s'):
            print(depth_image_3d.shape, color_image.shape)

            # number += 1
            # out_rgb = f'D:\innovision2022\RGBD_data\marker-detection\{number}_rgb.png'
            # out_dep = f'D:\innovision2022\RGBD_data\marker-detection\{number}_depth.png'

            # cv2.imwrite(out_rgb, frame_markers)
            # save_depth(out_dep, depth_image)

        # if curr_frame > 100 and curr_frame % 40 == 10:
        if key == ord('e'):
            # get April tags mid point position
            midpoints = np.zeros((4, 3))
            for i in range(len(ids)):
                idx = ids[i][0] - 1
                c = corners[i][0]
                mid_x = round(c[:, 0].mean())
                mid_y = round(c[:, 1].mean())
                mid_dep = aligned_depth_frame.get_distance(mid_x, mid_y)
                depth_point = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [mid_x, mid_y], mid_dep)
                midpoints[idx] = [depth_point[0], depth_point[1], mid_dep]

            # match the April tags ID with PSCS ID

            [l2, l3, l4, l1] = midpoints
            print(l1, l2, l3, l4)

            # calculate inverse transformation
            T = get_tp(l1, l2, l3, l4)
            inv_T = np.linalg.inv(T)
            print(np.round(np.dot(inv_T, [l2[0], l2[1], l2[2], 1]), 8))

            # pcd.transform([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            pcd.transform(inv_T)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                origin=l2)
            o3d.visualization.draw_geometries([pcd, mesh_frame])

            # torso_length = abs(l1[0] - l2[0])
            # print(torso_length)

            # N = 100  # num of cross-sections
            # alpha = torso_length / N  # thickness
            # BSR = []

            # xyz_load = np.asarray(pcd.points)
            # pts_groups = get_points_group(xyz_load, N, torso_length)

            # for level in range(N):
            #     theta = get_BSR_fit(pts_groups, level, draw=True)
            #     BSR.append(theta)

        curr_frame += 1

finally:
    pipeline.stop()


if __name__ == "__main__":
    pass

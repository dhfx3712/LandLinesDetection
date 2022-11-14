#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def plot_contrast_imgs(origin_img, converted_img, origin_img_title="origin_img", converted_img_title="converted_img",
                       converted_img_gray=False):
    """
    用于对比显示两幅图像
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
    ax1.set_title(origin_img_title)
    ax1.imshow(origin_img)
    ax2.set_title(converted_img_title)
    if converted_img_gray == True:
        ax2.imshow(converted_img, cmap="gray")
    else:
        ax2.imshow(converted_img)
    plt.show()


# 定义棋盘横向和纵向的角点个数
nx = 9
ny = 6


def cal_calibrate_params(file_paths):
    """
    计算相机校准参数,把对象点的坐标和检测的角点坐标一一对应
    """

    object_points = []
    image_points = []
    # 对象点的坐标
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 检测每幅图像角点坐标
    for file_path in file_paths:
        img = mpimg.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if rect == True:
            object_points.append(objp)
            image_points.append(corners)
    # 获取畸变系数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def img_undistort(img, mtx, dist):
    """
    图像去畸变
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

# 测试去畸变函数的效果
file_paths = glob.glob("./camera_cal/calibration*.jpg")
ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(file_paths)
# print (mtx)
if mtx.all() != None:
    img = mpimg.imread("./test_images/test1.jpg")
    undistort_img = img_undistort(img, mtx, dist)
    print("two pic")
    plot_contrast_imgs(img, undistort_img)
    print("done!")
else:
    print("failed")





def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    """
    使用sobel和选择颜色空间对图像进行处理
    """

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Threshold x gradient
    sxbinarx = np.zeros_like(scaled_sobel)
    sxbinarx[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    color_binary = np.zeros_like(sxbinary)
    color_binary[((sxbinary == 1) | (s_binary == 1)) & (l_channel > 100)] = 1
    return color_binary


# 测试使用sobel和选择颜色空间
img = mpimg.imread("./test/frame45.jpg")
result = pipeline(img)
plot_contrast_imgs(img, result, converted_img_gray=True)




def cal_perspective_params(img, points):
    """
    计算透视变换参数
    """

    offset_x = 330
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y],
                      [img_size[0] - offset_x, img_size[1] - offset_y]
                      ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse


def img_perspect_transform(img, M):
    '''
    图像透视变换
    '''

    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)


# 获取透视变换参数,每一个角度拍摄的透视变换矩阵都不一样
# 但是车辆的摄像头的位置固定，角度是固定的，用同一个透视转换矩阵就可以不用重复计算
img = mpimg.imread("./test_images/straight_lines2.jpg")
img = cv2.line(img, (601, 448), (683, 448), (255, 0, 0), 3)
img = cv2.line(img, (683, 448), (1097, 717), (255, 0, 0), 3)
img = cv2.line(img, (1097, 717), (230, 717), (255, 0, 0), 3)
img = cv2.line(img, (230, 717), (601, 448), (255, 0, 0), 3)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("./test_images/straight_lines2_line.jpg", img)
undistort_img = img_undistort(img, mtx, dist)
points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
M, M_inverse = cal_perspective_params(img, points)

# 测试透视图变换
if M != None:
    origin_img = mpimg.imread("./test_images/straight_lines2_line.jpg")
    undistort_img = img_undistort(origin_img, mtx, dist)
    transform_img = img_perspect_transform(undistort_img, M)
    plot_contrast_imgs(origin_img, transform_img, converted_img_gray=True)
else:
    print("transform failed!")


def cal_line_param(binary_warped):
    """
    binary_warped：图像处理过后的二值图像
    """

    # 图像的下半部分直方图统计
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # 设置滑动窗口的数量，计算每一个窗口的高度
    nwindows = 9
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # 获取图像中不为0的点
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 车道检测的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 设置x的检测范围
    margin = 100
    # 设置最小像素点
    minpix = 50
    # 记录检测出的左右车道点
    left_lane_inds = []
    right_lane_inds = []

    # 遍历该副图像中的每一个窗口
    for window in range(nwindows):
        # 设置窗口的x,y的检测范围
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 获得在窗口内部，且不为0的点
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，把该点集合的平均值设置当前点
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # 把检测出所有左右车道点分别进行合并
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 在图像中获取这些点
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 用曲线拟合检测出的点
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit



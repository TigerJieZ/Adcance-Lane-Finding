import numpy as np
import cv2
import pickle
import line
import calibration
from line import Line


def sobel_xy(image, kernel):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel), \
           cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)


def direction_gradient(image, kernel, threshold):
    sobel_x, sobel_y = sobel_xy(image, kernel)
    abs_gradient_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = threshold_filter(abs_gradient_dir, threshold)
    return binary_output.astype(np.uint8)


def m_sobel(image, threshold, transverse_axis='x'):
    if transverse_axis is 'x':
        sobel_image = abs(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    elif transverse_axis is 'y':
        sobel_image = abs(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    else:
        print('Please enter the correct Sobel direction')
        return None
    scaled_sobel = np.uint8(255 * sobel_image / np.max(sobel_image))

    temp = threshold_filter(scaled_sobel, threshold)

    return temp


def magnitude_gradient(image, kernel, threshold):
    sobel_x, sobel_y = sobel_xy(image, kernel)
    # Calculated gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 转换成unit8
    scale_factor = np.max(gradient_magnitude) / 255
    gradient_magnitude = (gradient_magnitude / scale_factor).astype(np.uint8)

    # The threshold is 0
    temp = threshold_filter(gradient_magnitude, threshold)

    return temp


def gradient_combine(image, threshold_x, threshold_y, threshold_mag, threshold_dir):
    sobel_x = m_sobel(image, threshold_x, 'x')
    sobel_y = m_sobel(image, threshold_y, 'y')
    mag_img = magnitude_gradient(image, 3, threshold_mag)
    dir_img = direction_gradient(image, 15, threshold_dir)

    # Combined gradient measurement
    gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
    gradient_comb[((sobel_x > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255

    # show_img(gradient_comb, 10000, 'gradient')

    return gradient_comb


def threshold_filter(ch, threshold=(80, 255)):
    binary = np.zeros_like(ch)
    # cv2.imshow('fda',binary)
    # cv2.waitKey(10000)
    binary[(ch > threshold[0]) & (ch <= threshold[1])] = 255
    return binary


def hls_combine(img, threshold_h, threshold_l, threshold_s):
    # The transformation of HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_data = hls[:, :, 0]
    l_data = hls[:, :, 1]
    s_data = hls[:, :, 2]

    h_img = threshold_filter(h_data, threshold_h)
    l_img = threshold_filter(l_data, threshold_l)
    s_img = threshold_filter(s_data, threshold_s)

    # Two cases - lane lines in the shadow
    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255  # | (R > 1)] = 255

    return hls_comb


def warp_image(img, src, dst, size):
    """ Perspective transformation """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    # show_img(warp_img, 10000, 'warp')

    return warp_img, M, Minv


def gaussian_blur(image):
    # Gauss fuzzy processing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blur_gray


def store_data(shape):
    s_LTop2, s_RTop2 = [shape[0] / 2 - 24, 5], [shape[0] / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, shape[1]], [shape[0] - 110, shape[1]]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    mtx, dist = calibration.calib()

    f = open('temp.pkl', 'wb')
    pickle.dump(mtx, f)
    pickle.dump(dist, f)
    pickle.dump(src, f)
    pickle.dump(dst, f)

    f.close()

    return mtx, dist, src, dst


def load_data():
    f = open('temp.pkl', 'rb')
    mtx = pickle.load(f)
    dist = pickle.load(f)
    src = pickle.load(f)
    dst = pickle.load(f)
    return mtx, dist, src, dst


def print_vehicle_data(image, left_line=Line(), right_line=Line()):
    '''Print vehicle location and lane radius information'''

    cv2.putText(image, 'Radius of Curvature = ' + str(round(left_line.radius_of_curvature, 3)) + '(m)',
                (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    center_lane = (right_line.startX - left_line.startX) / 2+left_line.startX
    lane_width = right_line.startX - left_line.startX
    xm_per_pix = 3.7 * (720 / 1280) / lane_width  # Pixel / meter
    center_car = 640 / 2
    if center_lane > center_car:
        deviation = 'Vehicle is ' + str(
            round(abs(center_lane - center_car) *xm_per_pix, 3)) + 'm left of center'
    elif center_lane < center_car:
        deviation = 'Vehicle is ' + str(
            round(abs(center_lane - center_car) *xm_per_pix, 3)) + 'm right of center'
    else:
        deviation = 'Center'
    left_line.deviation = deviation
    cv2.putText(image, deviation, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    return image

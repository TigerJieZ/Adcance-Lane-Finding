from utils import *
import m_utils
from line import Line
import os
import time
from moviepy.editor import *


def run(file):
    # If the camera correction data and the perspective data file do not exist, the data is calculated and the data is recorded otherwise.
    if not os.path.exists('temp.pkl'):
        mtx, dist, src, dst = m_utils.store_data([640, 128])
    else:
        mtx, dist, src, dst = m_utils.load_data()
    camera = cv2.VideoCapture(file)
    left_line = Line()
    right_line = Line()
    out = cv2.VideoWriter('video/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, (640, 680))
    
    while True:
        _, image = camera.read()
        # show_img(cv2.resize(image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA),10000,'image')

        # Correction of camera distortion
        undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
        # show_img(undistorted_image,10000,'undistorted')

        # The image is doubling
        zoom_image = cv2.resize(undistorted_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        # show_img(zoom_image, 100000, 'zoom')

        # Gauss vagueness
        # gaussian_image = m_utils.gaussian_blur(zoom_image)
        # show_img(gaussian_image, 10000, 'gaussian')

        # Intercept the ROI area (the picture contains the road)
        shape = undistorted_image.shape
        roi_image = zoom_image[220:shape[0] - 12, 0:shape[1], 2]
        roi_color_image = zoom_image[220:shape[0] - 12, 0:shape[1]]
        # show_img(roi_color_image, 100000, 'roi')

        # ----------gradient descent------------
        # Gradient descent with red channel
        gradient_image = m_utils.gradient_combine(roi_image,
                                                  (35, 100),
                                                  (30, 255),
                                                  (30, 255),
                                                  (0.7, 1.3))
        # show_img(gradient_image, 10000, 'gradient')
        # HLS color spatial gradient descent
        hls_image = m_utils.hls_combine(roi_color_image, (10, 100), (0, 60), (85, 255))
        # show_img(hls_image, 10000, 'hls')

        # Recombine the images of HLS and gradient descent
        result = np.zeros_like(gradient_image).astype(np.uint8)
        result[(gradient_image > 1)] = 100
        result[(hls_image > 1)] = 255
        # show_img(result, 10000, 'result')

        # import pylab
        # pylab.imshow(result)
        # pylab.plot([110, 296, 530, 344], [180, 5, 180, 5], 'r*')
        # # pylab.show()

        # The perspective image makes the image look down at the angle of view
        perspective_image, M, Minv = warp_image(result, src=src, dst=dst, size=(720, 720))
        # perspective_image_color=[]
        # time_=time.time()
        # for row in perspective_image:
        #     temp=[]
        #     for point in row:
        #         temp.append([point,point,point])
        #     perspective_image_color.append(temp)
        # print(time.time()-time_)
        perspective_image_color=np.dstack((perspective_image,perspective_image,perspective_image))

        # zoom_perspective_image = cv2.resize(perspective_image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        # show_img(perspective_image, 10000, 'perspective')

        # Looking for lane lines
        try:
            search_image = find_lr_lines(perspective_image, left_line=left_line, right_line=right_line)
        except Exception as e:
            left_line.detected = False
            print(e)

        # show_img(search_image,100000,'search')
        # Draw a lane line
        w_color_result = draw_lane(search_image, left_line, right_line)
        # show_img(w_comb_result, 10000, 'w_comb')
        # show_img(w_color_result, 10000, 'w_color')

        # Perspective perspective of drawing lane lines into the original angle
        color_road = cv2.warpPerspective(w_color_result, Minv, (result.shape[1], result.shape[0]))
        # show_img(color_road, 10000, 'color_road')

        # Superposition the lane marking of the perspective back to the original image into the original image
        mask = np.zeros_like(zoom_image)
        mask[220:shape[0] - 12, 0:shape[1]] = color_road
        road_image = cv2.addWeighted(zoom_image, 1, mask, 0.3, 0)
        # show_img(road_image, 10000, 'result')

        # Print vehicle location information
        info_road = m_utils.print_vehicle_data(image=road_image, left_line=left_line, right_line=right_line)

        # Parallel visualization
        debug_image=np.hstack((cv2.resize(np.array(perspective_image_color),(320,320)),cv2.resize(search_image,(320,320))))
        # show_img(debug_image,10000,'debug')
        debug_image=np.vstack((debug_image,info_road))

        show_img(debug_image,10,'road')
        out.write(debug_image)
    out.release()
    camera.release()

def process_image(image):
    # 相机失真的矫正
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    # show_img(undistorted_image,10000,'undistorted')

    # Image doubled
    zoom_image = cv2.resize(undistorted_image, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    # show_img(zoom_image, 100000, 'zoom')

    # Gaussian Blur
    # gaussian_image = m_utils.gaussian_blur(zoom_image)
    # show_img(gaussian_image, 10000, 'gaussian')

    # Cut out the roi area (the picture contains the road)
    shape = undistorted_image.shape
    roi_image = zoom_image[220:shape[0] - 12, 0:shape[1], 2]
    roi_color_image = zoom_image[220:shape[0] - 12, 0:shape[1]]
    # show_img(roi_color_image, 100000, 'roi')

    # ----------梯度下降------------
    # Gradient decrease with red channel
    gradient_image = m_utils.gradient_combine(roi_image,
                                              (35, 100),
                                              (30, 255),
                                              (30, 255),
                                              (0.7, 1.3))
    # show_img(gradient_image, 10000, 'gradient')
    # hls color space gradient down
    hls_image = m_utils.hls_combine(roi_color_image, (10, 100), (0, 60), (85, 255))
    # show_img(hls_image, 10000, 'hls')

    # Combine images resulting from hls and gradient descent
    result = np.zeros_like(gradient_image).astype(np.uint8)
    result[(gradient_image > 1)] = 150
    result[(hls_image > 1)] = 255
    # show_img(result, 10000, 'result')

    # import pylab
    # pylab.imshow(result)
    # pylab.plot([110, 296, 530, 344], [180, 5, 180, 5], 'r*')
    # # pylab.show()

    # 透视图像使图像呈俯视角度
    perspective_image, M, Minv = warp_image(result, src=src, dst=dst, size=(720, 720))
    # perspective_image_color=[]
    # time_=time.time()
    # for row in perspective_image:
    #     temp=[]
    #     for point in row:
    #         temp.append([point,point,point])
    #     perspective_image_color.append(temp)
    # print(time.time()-time_)
    perspective_image_color = np.dstack((perspective_image, perspective_image, perspective_image))

    # zoom_perspective_image = cv2.resize(perspective_image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    # show_img(perspective_image, 10000, 'perspective')

    # Find lane lines
    try:
        search_image = find_lr_lines(perspective_image, left_line=left_line, right_line=right_line)
    except Exception as e:
        left_line.detected = False
        search_image=perspective_image_color
        print(e)

    # show_img(search_image,100000,'search')
    # Draw lane lines
    w_color_result = draw_lane(search_image, left_line, right_line)
    # show_img(w_comb_result, 10000, 'w_comb')
    # show_img(w_color_result, 10000, 'w_color')

    # Will draw the lane line perspective of the original perspective perspective
    color_road = cv2.warpPerspective(w_color_result, Minv, (result.shape[1], result.shape[0]))
    # show_img(color_road, 10000, 'color_road')

    # Overlap the lane mark of the perspective of the original angle to the original image
    mask = np.zeros_like(zoom_image)
    mask[220:shape[0] - 12, 0:shape[1]] = color_road
    road_image = cv2.addWeighted(zoom_image, 1, mask, 0.3, 0)
    # show_img(road_image, 10000, 'result')

    # Print vehicle location information
    info_road = m_utils.print_vehicle_data(image=road_image, left_line=left_line, right_line=right_line)

    # Parallel visualization
    debug_image = np.hstack(
        (cv2.resize(np.array(perspective_image_color), (320, 320)), cv2.resize(search_image, (320, 320))))
    # show_img(debug_image,10000,'debug')
    debug_image = np.vstack((debug_image, info_road))
    show_img(debug_image,10,'debug')

    return debug_image

if __name__ == '__main__':
    # If the camera correction data and the perspective of the data file does not exist
    # then calculate the data and record otherwise load the data
    if not os.path.exists('temp.pkl'):
        mtx, dist, src, dst = m_utils.store_data([640, 128])
    else:
        mtx, dist, src, dst = m_utils.load_data()
    left_line = Line()
    right_line = Line()
    # run('project_video.mp4')
    dir='video/'
    file='challenge_video.mp4'
    white_output = dir+'output_'+file
    clip1 = VideoFileClip(dir+file)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

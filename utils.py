import numpy as np
import cv2


def show_img(image, time, windowName):
    cv2.imshow(winname=windowName, mat=image)
    cv2.waitKey(time)


def warp_image(img, src, dst, size):
    """ Perspective transformation """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    # show_img(warp_img, 10000, 'warp')

    return warp_img, M, Minv


def rad_of_curvature(left_line, right_line):
    """ Measuring radius of curvature  """

    ploty = left_line.ally
    leftx, rightx = left_line.allx, right_line.allx

    leftx = leftx[::-1]  # Reverse matching up and down on Y
    rightx = rightx[::-1]  # Reverse matching up and down on Y

    # Definition of conversion from pixel space to rice in X and Y
    width_lanes = abs(right_line.startX - left_line.startX)
    ym_per_pix = 30 / 720  # 像素/米
    xm_per_pix = 3.7 * (720 / 1280) / width_lanes  # 像素/米
    # The defined value, here we want the radius of curvature
    # Maximum value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # X and Y fitting of new polynomials in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculating new radius of curvature
    left_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Radius of curvature
    left_line.radius_of_curvature = left_curvature
    right_line.radius_of_curvature = right_curvature


def smoothing(lines, pre_lines=3):
    # Collection line and print average line
    lines = np.squeeze(lines)
    avg_line = np.zeros(720)

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def blind_search(img, line_left, line_right):
    """
    Blind search - the first frame / lost lane line
    Using histogram and sliding window
    """
    # Get the histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    # Create an output image to draw and visualize the results。
    output = np.dstack((img, img, img)) * 255
    # show_img(output, 10000, 'output')

    # Find the vertices of the left and right sides of the histogram.
    # This will be the starting point of the left and right lines.
    start_left_x = np.argmax(histogram[:int(histogram.shape[0] / 2)])
    start_right_x = np.argmax(histogram[int(histogram.shape[0] / 2):]) + int(histogram.shape[0] / 2)

    # Select the number of sliding windows
    num_windows = 9
    # Setting window height
    window_height = int(img.shape[0] / num_windows)

    # Identify all non - zero pixels in X and Y positions in the image.
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # The current position of each window update
    current_left_x = start_left_x
    current_right_x = start_right_x

    ###################
    # Setting up the minimum number of pixels to find the wake-up window
    min_num_pixel = 50

    # Create an empty list to receive the left and right row pixel index
    win_left_lane = []
    win_right_lane = []

    window_margin = line_left.window_margin

    # Step by step slide window
    for window in range(num_windows):
        # Identifying the window boundaries of X and Y (right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_leftx_min = current_left_x - window_margin
        win_leftx_max = current_left_x + window_margin
        win_rightx_min = current_right_x - window_margin
        win_rightx_max = current_right_x + window_margin

        # Drawing Windows on visual images
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the non zero pixel in the X and Y in the window.
        left_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_leftx_min) & (
            nonzero_x <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_rightx_min) & (
            nonzero_x <= win_rightx_max)).nonzero()[0]
        # Attach these indexes to the list
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you find minpix pixels, wake the next window in its average position
        if len(left_window_inds) > min_num_pixel:
            current_left_x = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_right_x = np.int(np.mean(nonzero_x[right_window_inds]))

    # An array that connects the index
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extraction of left and right row pixels
    left_x, left_y = nonzero_x[win_left_lane], nonzero_y[win_left_lane]
    right_x, right_y = nonzero_x[win_right_lane], nonzero_y[win_right_lane]

    output[left_y, left_x] = [255, 0, 0]
    output[right_y, right_x] = [0, 0, 255]

    # Fitting two order polynomials to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    line_left.current_fit = left_fit
    line_right.current_fit = right_fit

    # Generate X and Y values for drawing.
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plot_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    line_left.prevX.append(left_plot_x)
    line_right.prevX.append(right_plot_x)

    if len(line_left.prevX) > 10:
        left_avg_line = smoothing(line_left.prevX, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        line_left.current_fit = left_avg_fit
        line_left.allx, line_left.ally = left_fit_plot_x, ploty
    else:
        line_left.current_fit = left_fit
        line_left.allx, line_left.ally = left_plot_x, ploty

    if len(line_right.prevX) > 10:
        right_avg_line = smoothing(line_right.prevX, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        line_right.current_fit = right_avg_fit
        line_right.allx, line_right.ally = right_fit_plot_x, ploty
    else:
        line_right.current_fit = right_fit
        line_right.allx, line_right.ally = right_plot_x, ploty

    line_left.startX, line_right.startX = line_left.allx[ - 1], line_right.allx[- 1]
    line_left.endX, line_right.endX = line_left.allx[0], line_right.allx[0]

    line_left.detected, line_right.detected = True, True
    # Print curvature radius
    rad_of_curvature(line_left, line_right)
    return output


def prev_window_refer(b_img, left_line, right_line):
    """
    When the lane line is detected in the last frame, refer to the front window information
    """
    # Create an output image to draw and visualize the results.
    output = np.dstack((b_img, b_img, b_img)) * 255

    # Identify all non - zero pixels in X and Y positions in the image.
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Setting the margin of the window
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the non zero pixel in the X and Y in the window.
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # Extraction of left and right row pixels
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # Fitting two order polynomials to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate X and Y values for drawing.
    plot_y = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_plot_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # left_x_avg = np.average(left_plot_x)
    # right_x_avg = np.average(right_plot_x)

    left_line.prevX.append(left_plot_x)
    right_line.prevX.append(right_plot_x)

    if len(left_line.prevX) > 10:
        left_avg_line = smoothing(left_line.prevX, 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plot_x, plot_y
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plot_x, plot_y

    if len(right_line.prevX) > 10:
        right_avg_line = smoothing(right_line.prevX, 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plot_x, plot_y
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plot_x, plot_y

    # 去blind_search如果车道线的标准值高。
    standard = np.std(right_line.allx - left_line.allx)

    if standard > 80:
        left_line.detected = False

    left_line.startX, right_line.startX = left_line.allx[len(left_line.allx) - 1], right_line.allx[
        len(right_line.allx) - 1]
    left_line.endX, right_line.endX = left_line.allx[0], right_line.allx[0]

    # Print curvature radius
    rad_of_curvature(left_line, right_line)
    return output

def blind_search_challenge(b_img, left_line, right_line):
    """
    blind search - first frame, lost lane lines
    using histogram & sliding window
    give different weight in color info(0.8) & gradient info(0.2) using weighted average
    """
    # Create an output image to draw on and  visualize the result
    # output = np.dstack((b_img, b_img, b_img)) * 255
    output = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)

    # Choose the number of sliding windows
    num_windows = 9
    # Set height of windows
    window_height = np.int(b_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_line.startx == None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(b_img[int(b_img.shape[0] * 2 / 3):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        start_leftX = np.argmax(histogram[:midpoint])
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        current_leftX = start_leftX
        current_rightX = start_rightX
    else:
        current_leftX = left_line.startx
        current_rightX = right_line.startx

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    left_weight_x, left_weight_y = [], []
    right_weight_x, right_weight_y = [], []
    window_margin = left_line.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[0] - window * window_height
        win_leftx_min = int(current_leftX - window_margin)
        win_leftx_max = int(current_leftX + window_margin)
        win_rightx_min = int(current_rightX - window_margin)
        win_rightx_max = int(current_rightX + window_margin)

        if win_rightx_max > 720:
            win_rightx_min = b_img.shape[1] - 2 * window_margin
            win_rightx_max = b_img.shape[1]

        # Draw the windows on the visualization image
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:

            win = b_img[win_y_low:win_y_high, win_leftx_min:win_leftx_max]
            temp, count_g, count_h = 0, 0, 0
            for i in range(win.shape[1]):
                for j in range(win.shape[0]):
                    if win[j, i] >= 70 and win[j, i] <= 130:
                        temp += 0.2 * (i + win_leftx_min)
                        count_g += 1
                        output[j + win_y_low, i + win_leftx_min] = (255, 0, 0)
                    elif win[j, i] > 220:
                        temp += 0.8 * (i + win_leftx_min)
                        count_h += 1
                        output[j + win_y_low, i + win_leftx_min] = (0, 0, 255)
                        # else:
                        #    output[j + win_y_low, i + win_leftx_min] = (255, 255, 255)
            if not (count_h == 0 and count_g == 0):
                left_w_x = temp / (0.2 * count_g + 0.8 * count_h)  # + win_leftx_min
                #cv2.circle(output, (int(left_w_x), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                #cv2.circle(output, (int(current_leftX), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                left_weight_x.append(int(left_w_x))
                left_weight_y.append(int((win_y_low + win_y_high) / 2))

                current_leftX = int(left_w_x)

        if len(right_window_inds) > min_num_pixel:

            win = b_img[win_y_low:win_y_high, win_rightx_min:win_rightx_max]
            temp, count_g, count_h = 0, 0, 0
            for i in range(win.shape[1]):
                for j in range(win.shape[0]):
                    if win[j, i] >= 70 and win[j, i] <= 130:
                        temp += 0.2 * (i + win_rightx_min)
                        count_g += 1
                        output[j + win_y_low, i + win_rightx_min] = (255, 0, 0)
                    elif win[j, i] > 200:
                        temp += 0.8 * (i + win_rightx_min)
                        count_h += 1
                        output[j + win_y_low, i + win_rightx_min] = (0, 0, 255)
                        # else:
                        #    output[j + win_y_low, i + win_rightx_min] = (255, 255, 255)
            if not (count_h == 0 and count_g == 0):
                right_w_x = temp / (0.2 * count_g + 0.8 * count_h)  # + win_leftx_min
                #cv2.circle(output, (int(right_w_x), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                #cv2.circle(output, (int(current_rightX), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                right_weight_x.append(int(right_w_x))
                right_weight_y.append(int((win_y_low + win_y_high) / 2))
                current_rightX = int(right_w_x)

    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    #output[lefty, leftx] = [255, 0, 0]
    #output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_weight_y, left_weight_x, 2)
    right_fit = np.polyfit(right_weight_y, right_weight_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    # frame to frame smoothing
    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_line.startx, right_line.startx = left_line.allx[len(left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    left_line.detected, right_line.detected = True, True
    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output

def prev_window_refer_challenge(b_img, left_line, right_line):
    """
    refer to previous window info - after detecting lane lines in previous frame
    give different weight in color info(0.8) & gradient info(0.2) using weighted average
    """
    # Create an output image to draw on and  visualize the result
    output = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)

    # Set margin of windows
    window_margin = left_line.window_margin

    left_weight_x, left_weight_y = [], []
    right_weight_x, right_weight_y = [], []

    temp, count_g, count_h = 0, 0, 0
    for i, j in enumerate(left_line.allx):
        for m in range(window_margin):
            j1, j2 = int(j) + m, int(j) - m

            if b_img[i, j1] >= 70 and b_img[i, j1] <= 130:
                temp += 0.2 * j1
                count_g += 1
                output[i, j1] = (255, 0, 0)
            if b_img[i, j2] >= 70 and b_img[i, j2] <= 130:
                temp += 0.2 * j2
                count_g += 1
                output[i, j2] = (255, 0, 0)
            if b_img[i, j1] > 220:
                temp += 0.8 * j1
                count_h += 1
                output[i, j1] = (0, 0, 255)
            if b_img[i, j2] > 220:
                temp += 0.8 * j2
                count_h += 1
                output[i, j2] = (0, 0, 255)
        if (i+1) % 80 == 0:
            if not (count_h == 0 and count_g == 0):
                left_w_x = temp / (0.2 * count_g + 0.8 * count_h)  # + win_leftx_min
                #cv2.circle(output, (int(left_w_x), (i+1-40)), 10, (255, 0, 0), -1)
                left_weight_x.append(int(left_w_x))
                left_weight_y.append((i+1-40))
            temp, count_g, count_h = 0, 0, 0

    temp, count_g, count_h = 0, 0, 0
    for i, j in enumerate(right_line.allx):
        if j >= 720 - (window_margin):
            for m in range(2*(window_margin)):
                k = 720 - 2*(window_margin) + m
                if b_img[i, k] >= 70 and b_img[i, k] <= 130:
                    temp += 0.2 * k
                    count_g += 1
                    output[i, k] = (255, 0, 0)
                if b_img[i, k] > 220:
                    temp += 0.8 * k
                    count_h += 1
                    output[i, k] = (0, 0, 255)
        else:
            for m in range(window_margin):
                j1, j2 = int(j) + m, int(j) - m
                if b_img[i, j1] >= 70 and b_img[i, j1] <= 130:
                    temp += 0.2 * j1
                    count_g += 1
                    output[i, j1] = (255, 0, 0)
                if b_img[i, j2] >= 70 and b_img[i, j2] <= 130:
                    temp += 0.2 * j2
                    count_g += 1
                    output[i, j2] = (255, 0, 0)
                if b_img[i, j1] > 220:
                    temp += 0.8 * j1
                    count_h += 1
                    output[i, j1] = (0,0, 255)
                if b_img[i, j2] > 220:
                    temp += 0.8 * j2
                    count_h += 1
                    output[i, j2] = (0, 0, 255)
        if (i + 1) % 80 == 0:
            if not (count_h == 0 and count_g == 0):
                right_w_x = temp / (0.2 * count_g + 0.8 * count_h)
                #cv2.circle(output, (int(right_w_x), (i+1-40)), 10, (255, 0, 0), -1)
                right_weight_x.append(int(right_w_x))
                right_weight_y.append((i+1-40))
            temp, count_g, count_h = 0, 0, 0

    #output[lefty, leftx] = [255, 0, 0]
    #output[righty, rightx] = [0, 0, 255]

    if len(left_weight_x) <= 5:
        left_weight_x = left_line.allx
        left_weight_y = left_line.ally
    if len(right_weight_x) <= 5:
        right_weight_x = right_line.allx
        right_weight_y = right_line.ally

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_weight_y, left_weight_x, 2)
    right_fit = np.polyfit(right_weight_y, right_weight_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    # frame to frame smoothing
    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    # goto blind_search if the standard value of lane lines is high.
    standard = np.std(right_line.allx - left_line.allx)

    if (standard > 80):
        left_line.detected = False

    left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output

def find_lr_lines(binary_img, left_line, right_line,isChallenge=False):
    """
    Look for left and right lines, isolate left and right lines
    Blind search - the first frame / lost lane line
    Last window - check the lane line in the previous frame
    """

    # If there is no lane information
    if left_line.detected is False:
        if isChallenge:
            return blind_search_challenge(binary_img, left_line, right_line)
        else:
            return blind_search(binary_img, left_line, right_line)
    # If there is a lane information
    else:
        if isChallenge:
            return prev_window_refer_challenge(binary_img, left_line, right_line)
        else:
            return prev_window_refer(binary_img, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(0, 0, 255), road_color=(0, 255, 0)):
    """ Draw lane lines and current drive space """
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plot_x, right_plot_x = left_line.allx, right_line.allx
    plot_y = left_line.ally

    # A polygon is generated to display the search window area.
    # The X and Y coordinates of the recast are available in the format of fillpoly () CV2.
    left_pts_l = np.array([np.transpose(np.vstack([left_plot_x - window_margin / 5, plot_y]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array([np.transpose(np.vstack([right_plot_x - window_margin / 5, plot_y]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plot_x + window_margin / 5, plot_y])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw a lane on a twisted blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # The X and Y points of the recast are the formats available for fillpoly () CV2.
    pts_left = np.array([np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x - window_margin / 5, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw a driving area
    cv2.fillPoly(window_img, np.int_([pts]), road_color)

    return window_img

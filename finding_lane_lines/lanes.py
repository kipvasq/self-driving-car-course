import cv2
import numpy as np
import matplotlib.pyplot as plt

def applyCanny(image):
    # grayscale image (y, x), no third dimension!
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # gaussian blur image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # canny image (derivative in all directions to determine gradient)
    canny_image = cv2.Canny(blurred_image, 50, 150) 

    return canny_image

def applyRegionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)] # triangle
    ])

    # zeroes image with same shape of image, and create mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    return cv2.bitwise_and(image, mask)

def applyHoughLines(image):
    # get hough lines with polar parameters
    hough_lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = np.zeros_like(image)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)

    if (hough_lines is not None):
        average_lines = averageSlopeIntercept(line_image, hough_lines)

        for line in average_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 155, 0), 5)

    return line_image

def averageSlopeIntercept(image, hough_lines):
    left_fit = []
    right_fit = []

    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]

        if (slope < 0):
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = createLineCoordinates(image.shape, left_fit_average)
    right_line = createLineCoordinates(image.shape, right_fit_average)

    return [left_line, right_line]

def createLineCoordinates(shape, line_params):
    slope, intercept = line_params
    y1 = shape[0]
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def applyLineOverlay(image, line_image):
    return cv2.addWeighted(image, 0.8, line_image, 1, 1)

video_capture = cv2.VideoCapture('test2.mp4')

while (video_capture.isOpened()):
    # read video, get current frame
    _, image = video_capture.read()
    lane_image = np.copy(image)

    try:
        # find lane lines
        canny_image = applyCanny(lane_image)
        regioned_image = applyRegionOfInterest(canny_image)
        hough_lines_image = applyHoughLines(regioned_image)
        combined_image = applyLineOverlay(image, hough_lines_image)
    except:
        break 

    # show image
    cv2.imshow("Current Frame", combined_image)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()

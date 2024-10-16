import cv2
import numpy as np


def lane_follower():
    # Read the image
    path = input()
    image = cv2.imread(path)
    if image is None:
        print("Error: Image not found or unable to read.")
        return

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the red color in HSV space

    lower_white = np.array([0, 0, 175])
    upper_white = np.array([150, 30, 255])
    lower_yellow = np.array([0, 0, 170])
    upper_yellow = np.array([30, 255, 255])

    # Create masks for the red color
    mask_new = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_new[: image.shape[0] // 2, :] = 0

    width = image.shape[1]
    mask_left = np.ones(mask_new.shape)
    mask_left[:, int(np.floor(width / 2)):width + 1] = 0
    mask_right = np.ones(mask_new.shape)
    mask_right[:, 0:int(np.floor(width / 2))] = 0

    def right_white():
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_white = cv2.GaussianBlur(mask_white, (5, 5), 0)
        # cv2.imshow('bla', mask_white)
        # cv2.waitKey(0)
        sobel_x = cv2.Sobel(mask_white, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_white, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
        threshold_value = 100
        _, mask_mag = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        mask_sobelx_pos = (sobel_x > 0)
        mask_sobelx_neg = (sobel_x < 0)
        mask_sobely_pos = (sobel_y > 0)
        mask_sobely_neg = (sobel_y < 0)
        mask_right_edge = mask_new * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg
        return mask_right_edge

    def left_yellow():
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
        sobel_x = cv2.Sobel(mask_yellow, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_yellow, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
        threshold_value = 100
        _, mask_mag = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        mask_sobelx_pos = (sobel_x > 0)
        mask_sobelx_neg = (sobel_x < 0)
        mask_sobely_pos = (sobel_y > 0)
        mask_sobely_neg = (sobel_y < 0)
        mask_left_edge = mask_new * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg
        return mask_left_edge

    mask_right = right_white()
    mask_left = left_yellow()
    # cv2.imshow('Detected Red Rectangles', mask_left)
    # cv2.waitKey(0)
    cv2.imshow('das', mask_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
lane_follower()

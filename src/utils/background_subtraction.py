import cv2
import numpy as np

class BackgroundSubtraction():
    GAUSS_MIXTURE = 'Gaussian Mixture'
    KNN = 'K-nearest neighbours'
    SKIN_COLOR_YCbCr = 'Skin Color YCbCr'
    SKIN_COLOR_HSV = 'Skin Color HSV'

    BACKGROUND_SUB_TYPE = GAUSS_MIXTURE
    BACKGROUND_SUB_THRESHOLD = 200  # background subtraction threshold
    DETECT_SHADOWS = False
    HISTORY = 2
    LEARNING_RATE = 0  # 1.0 / history

    blur_size = (15, 15)
    blur_gaussian_size = (3, 3)

    def __init__(self):
        self._bg_model = None
        self._sub_type = None
        self._bg_sub_threshold = None
        self._detect_shadows = None
        self._history = None
        self._learning_rate = None  # 1.0 / history
        self._is_bg_captured = False

    def resetBackgroundModel(self):
        self._bg_model = None
        self._is_bg_captured = False

    def create_background_model_gmm(self, history, bg_sub_threshold, detect_shadows):
        # https://www.authentise.com/post/segment-background-using-computer-vision
        # Gaussian Mixture-based Background/Foreground Segmentation Algorithm
        self._bg_model = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=bg_sub_threshold, detectShadows=detect_shadows)
        self._is_bg_captured = True
        return self._bg_model

    def create_background_model_knn(self, history, bg_sub_threshold, detect_shadows):
        # https://www.authentise.com/post/segment-background-using-computer-vision
        self._bg_model = cv2.createBackgroundSubtractorKNN(history=history, varThreshold=bg_sub_threshold, detectShadows=detect_shadows)
        self._is_bg_captured = True
        return self._bg_model

    def createBackgroundModel(self, sub_type, history, bg_sub_threshold, detect_shadows):
        if not self._is_bg_captured:
            self._sub_type = sub_type

            if self._sub_type == self.GAUSS_MIXTURE:
                return self.create_background_model_gmm(history, bg_sub_threshold, detect_shadows)
            elif self._sub_type == self.KNN:
                return self.create_background_model_knn(history, bg_sub_threshold, detect_shadows)
            elif self._sub_type == self.SKIN_COLOR_YCbCr:
                return None
            elif self._sub_type == self.SKIN_COLOR_HSV:
                return None

        return None

    # Apply GaussianBlur filter to the image and remove the background using MOG/KNN algorithm
    def removeBackground(self, img, learning_rate, apply_filter=False):
        # return self.opcvsub.remove_background(frame, learning_rate)
        fg_mask = None
        if self._bg_model is not None:
            if apply_filter:
                # Apply GaussianBlur filter to the image
                blur = cv2.GaussianBlur(img, (3, 3), 0)
            else:
                blur = img

            if self._sub_type == self.GAUSS_MIXTURE or self._sub_type == self.KNN:
                # Gaussian Mixture-based Background/Foreground Segmentation
                fg_mask = self._bg_model.apply(blur, learningRate=learning_rate)
        return fg_mask

    def skinFilterForYCrCbSpace(self, img):
        # Converting from gbr to YCbCr color space to filter skin color
        # Skin color ranges for YCbCr
        lower_range = np.array([0, 135, 85], dtype=np.uint8)
        upper_range = np.array([255, 180, 135], dtype=np.uint8)
        # Skin color segmentation for YCbCr space
        imgYCrCb = self.skinFilterForSpaceColor(img, lower_range, upper_range, cv2.COLOR_BGR2YCrCb)
        imgBGR = cv2.cvtColor(imgYCrCb, cv2.COLOR_YCrCb2BGR)

        return imgBGR

    def skinFilterForHSVSpace(self, img):
        # Converting from gbr to HSV color space to filter skin color
        # Skin color ranges for HSV space
        lower_range = np.array([0, 15, 0], dtype=np.uint8)
        upper_range = np.array([17, 170, 255], dtype=np.uint8)
        # Skin color segmentation for HSV space
        imgHSV = self.skinFilterForSpaceColor(img, lower_range, upper_range, cv2.COLOR_BGR2HSV)
        imgBGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

        return imgBGR

    def skinFilterForSpaceColor(self, img, lower_range, upper_range, spaceColor):
        # Converting from gbr to HSV/YCbCr color space
        conv_img = cv2.cvtColor(img, spaceColor)
        # Ckin color range for HSV/YCbCr color space
        mask = cv2.inRange(conv_img, lower_range, upper_range)
        # Noise remove
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        result = cv2.bitwise_and(conv_img, conv_img, mask=mask)
        #result = cv2.bitwise_not(_mask)
        return result
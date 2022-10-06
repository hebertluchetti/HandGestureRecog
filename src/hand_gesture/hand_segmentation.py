import cv2

from src.utils.finger import Finger
from src.utils.hand import Hand
from src.hand_gesture.base_segmentation import BaseSegmentation

# Environment:
# OS    : Windows 10
# OpenCV: 4.2.0
# Python: 3.7.7 (default, May  6 2020, 11:45:54) [MSC v.1916 64 bit (AMD64)]
# Version info: sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)

class HandSegmentation(BaseSegmentation):

    def __init__(self, source=0):
        super().__init__(source)
        self._finger_proc = Finger()
        self._blur_value = 41  # GaussianBlur parameter
        self._setting_window = 'Settings'

    def nothing(self, x):
        pass

    def changeThreshold(self, thr):
        self.resetBackgroundModel()
        print("Changed subtraction threshold to " + str(thr))

    def printThreshold(self, thr):
        print("Changed threshold to " + str(thr))

    def settingsWindow(self):
        cv2.namedWindow("Settings")
        cv2.resizeWindow("Settings", 640, 400)

        cv2.createTrackbar('bgSubThreshold', self._setting_window, self._bg_sub_threshold, 1000, self.changeThreshold)

        # Skin color ranges for YCbCr
        # lower_range = np.array([0, 135, 85], dtype=np.uint8)
        # upper_range = np.array([255, 180, 135], dtype=np.uint8)

        cv2.createTrackbar("Y Low", "Settings", 0, 255, self.nothing)
        cv2.createTrackbar("Y Up", "Settings", 255, 255, self.nothing)
        cv2.createTrackbar("Cb Low", "Settings", 135, 255, self.nothing)
        cv2.createTrackbar("Cb Up", "Settings", 180, 255, self.nothing)
        cv2.createTrackbar("Cr Low", "Settings", 85, 255, self.nothing)
        cv2.createTrackbar("Cr Up", "Settings", 135, 255, self.nothing)

        cv2.setTrackbarPos("H Low", "Settings", 0)
        cv2.setTrackbarPos("H Up", "Settings", 17)
        cv2.setTrackbarPos("S Low", "Settings", 15)
        cv2.setTrackbarPos("S Up", "Settings", 170)
        cv2.setTrackbarPos("V Low", "Settings", 0)
        cv2.setTrackbarPos("V Up", "Settings", 255)

        # Skin color ranges for HSV space
        # lower_range = np.array([0, 15, 0], dtype=np.uint8)
        # upper_range = np.array([17, 170, 255], dtype=np.uint8)

        # cv2.createTrackbar("H Low", "Settings", 0, 180, nothing)
        # cv2.createTrackbar("H Up", "Settings", 0, 180, nothing)
        # cv2.createTrackbar("S Low", "Settings", 0, 255, nothing)
        # cv2.createTrackbar("S Up", "Settings", 0, 255, nothing)
        # cv2.createTrackbar("V Low", "Settings", 0, 255, nothing)
        # cv2.createTrackbar("V Up", "Settings", 0, 255, nothing)
        #
        # cv2.setTrackbarPos("H Low", "Settings", 0)
        # cv2.setTrackbarPos("H Up", "Settings", 17)
        # cv2.setTrackbarPos("S Low", "Settings", 15)
        # cv2.setTrackbarPos("S Up", "Settings", 170)
        # cv2.setTrackbarPos("V Low", "Settings", 0)
        # cv2.setTrackbarPos("V Up", "Settings", 255)

    def releaseAll(self):
        self._cap_conf.releaseCapture()
        cv2.destroyAllWindows()

    def processVideo(self):
        # Camera
        self._cap = self._cap_conf.getCapture()
        self._is_bg_captured = False  # bool, whether the background captured
        self.bg_sub = None
        draw = True
        is_count_fingers = False
        main_win_name = "Original"
        msg = self.BACKGROUND_MSG
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Window Settings for keep the values
        self.settingsWindow()

        # Defines the area of interest
        (x_min, x_max, y_min, y_max, area, roi_color), width, height = self._cap_conf.defineHandROI()
        # Define the message position on the screen
        x_ref, y_ref = int(width * 0.005), int(height * 0.025)
        # Define the class message position on the screen
        x_class, y_class = x_ref, y_ref * 2

        hand = Hand()
        hand.setDimensions(x_min, y_min, x_max, y_max)

        try:
            while self._cap.isOpened():
                # Capture frame-by-frame
                ret, frame = self._cap.read()

                if not ret:
                    continue

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # Define _area to capture the hand movement
                self.drawHandROI(frame, x_min, x_max, y_min, y_max, roi_color)

                if (is_count_fingers and not self._is_bg_captured) or not is_count_fingers:
                    cv2.putText(frame, msg, (x_ref, y_ref), 1, 1.4, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, self.QUIT_MSG, (x_class, y_class), 1, 1.4, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(main_win_name, frame)

                #  Main operation
                if self._is_bg_captured:  # this part wont run until background captured

                    # ------------------------------------------------------------------------------------------------------
                    #  1-) Background subtraction operation
                    # ------------------------------------------------------------------------------------------------------
                    if self._is_bg_captured:  # this part wont run until background captured
                        result, valid_hand, hand_img, _ = self.getHandImage(hand, frame, x_min, y_min, x_max, y_max,
                                                                            draw)
                        if not result:
                            continue
                        cv2.imshow('Foreground', hand_img)

                    # 2-) Count the number of fingers
                    ####################################################################################################
                    # Draw the Hand ConvexHull and count the number of fingers
                    if is_count_fingers:
                        self._finger_proc.countFingers(frame, main_win_name, hand, draw)

                    # 4-) Draw Hand Contour to test some features
                    ####################################################################################################
                    # if circle_center is not None:
                        # Draw the finger numbers from hand convex hull
                        # self._handfinger_proc.drawFingers(frame, max_contour, _y_min, _y_max, _x_min, _x_max, circle_center, circle_offset)

                        # Draw Hand Contour
                        # self._contour_proc.drawHandContour(detect_hand, contours, circle_center, circle_offset)

                # Keyboard OP
                key = cv2.waitKey(27)

                if key == 27:  # press ESC to exit
                    break
                elif key == ord('b'):  # press 'b' to capture the background
                    if not self._is_bg_captured:
                        self._bg_sub_threshold = cv2.getTrackbarPos('bgSubThreshold', self._setting_window)
                        self.createBackgroundModel(self._bg_sub_type, self._history, self._bg_sub_threshold, self._detect_shadows, self._learning_rate)
                        msg = self.RESET_BACKGROUND_MSG
                        print('Background Model created !')
                    else:
                        print('Background Model already created !')
                elif key == ord('r'):  # press 'r' to reset the background
                    self.resetBackgroundModel()
                    msg = self.BACKGROUND_MSG
                    print('Reset Background !')
        finally:
            self.releaseAll()


def main():
    hand = HandSegmentation(0)
    hand.processVideo()


if __name__ == '__main__':
    main()

import cv2
import copy
import sys

from src.utils.pos_process import PosProcess
from src.utils.finger import Finger
from src.utils.capture_config import CaptureConfig
from src.utils.background_subtraction import BackgroundSubtraction
from src.utils.contour import Contour
from src.utils.hand import Hand

# Environment:
# OS    : Windows 10
# OpenCV: 4.2.0
# Python: 3.7.7 (default, May  6 2020, 11:45:54) [MSC v.1916 64 bit (AMD64)]
# Version info: sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)


class FingerDetection:
    def __init__(self, source=0):
        print('OpenCV: {}'.format(cv2.__version__))
        print('Python: {}'.format(sys.version))
        print('Version info: {}'.format(sys.version_info))

        self._init(source)

    def _init(self, source):
        self._cap_conf = CaptureConfig(source)
        self._cap = self._cap_conf.getCapture()
        self._pos_proc = PosProcess()
        self._finger_proc = Finger()
        self._bg_sub = BackgroundSubtraction()
        self._contour_proc = Contour()

        # variables
        self._is_bg_captured = False  # bool, whether the background captured
        self._bg_sub_type = BackgroundSubtraction.GAUSS_MIXTURE
        self._bg_sub_threshold = BackgroundSubtraction.BACKGROUND_SUB_THRESHOLD  # background subtraction threshold
        self._detect_shadows = BackgroundSubtraction.DETECT_SHADOWS
        self._history = BackgroundSubtraction.HISTORY
        self._learning_rate = BackgroundSubtraction.LEARNING_RATE  # 1.0 / history
        self.yellow_color = (0, 255, 255)

    def nothing(self, x):
        pass

    def changeThreshold(self, thr):
        self.resetBackgroundModel()
        print("Changed subtraction threshold to " + str(thr))

    def printThreshold(self, thr):
        print("Changed threshold to " + str(thr))

    def drawHandROI(self, frame, x_min, x_max, y_min, y_max, roi_color):
        # roi = frame[50:300, 380:600]
        # roi = roi[_y_min:_y_max, xMin:_x_max]  # clip the ROI
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), roi_color, 4)

    def createBackgroundModel(self, sub_type, history, bg_sub_threshold, detect_shadows, learning_rate):
        self._bg_sub.createBackgroundModel(sub_type, history, bg_sub_threshold, detect_shadows)

    def resetBackgroundModel(self):
        self._is_bg_captured = False
        self._bg_sub.resetBackgroundModel()

    # Create the _mask removing from the frame the background by using MOG/KNN segmentation algorithm
    # and remove the noise from this _mask applying morphological filters
    def removeBackground(self, img, learning_rate):
        # 1-) Create the _mask from the Gaussian Mixture-based Background/Foreground Segmentation
        # Apply GaussianBlur filter to the image and remove the background using MOG/KNN segmentation algorithm
        fg_mask = self._bg_sub.removeBackground(img, learning_rate)

        if fg_mask is not None:
            # 2-) Remove the noise from the _mask applying opening and closing morphological filter and GaussianBlur filter
            filtered_mask = self._pos_proc.removeMaskNoise(fg_mask)
            return filtered_mask

        return None

    def processVideo(self):
        # Camera
        cap = self._cap
        self._is_bg_captured = False  # bool, whether the background captured
        self.bg_sub = None
        is_count_fingers = True
        main_win_name = "Original"
        msg = "Press 'B' and wait 5 secs to show the hand"
        (x_min, x_max, y_min, y_max, roi_area, roi_color), _, _ = self._cap_conf.defineHandROI()
        hand = Hand()
        hand.setDimensions(x_min, y_min, x_max, y_max)

        try:
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    continue

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # Define _area to capture the hand movement
                self.drawHandROI(frame, x_min, x_max, y_min, y_max, roi_color)

                if (is_count_fingers and not self._is_bg_captured) or not is_count_fingers:
                    x = int(self._cap_conf.width * 0.1)
                    y = int(self._cap_conf.height * 0.05)
                    cv2.putText(frame, msg, (x, y), 1, 1.4, self.yellow_color, 1, cv2.LINE_AA)
                    cv2.imshow(main_win_name, frame)

                #  Main operation
                if self._is_bg_captured:  # this part wont run until background captured
                    # Taking a copy of the image
                    img = copy.deepcopy(frame)
                    roi = img[y_min:y_max, x_min:x_max]  # clip the ROI

                    # 1-) Create the _mask removing from the frame the background by using MOG/KNN segmentation algorithm
                    #     and remove the noise from this _mask applying morphological filters
                    ####################################################################################################
                    fg_mask = self.removeBackground(roi, self._learning_rate)
                    hand.setMask(fg_mask)

                    if fg_mask is None:
                        continue

                    # 2-) Find the contours in the image an select the biggest one
                    ####################################################################################################
                    _, max_contour, _ = self._contour_proc.findContours(fg_mask)
                    hand.setContour(max_contour)

                    # 3-) Count the number of fingers
                    ####################################################################################################
                    if is_count_fingers:
                        fingers = self._finger_proc.countFingers(frame, main_win_name, hand, True)
                        print("fingers =", fingers)

                # Keyboard OP
                key = cv2.waitKey(27)
                if key == 27 or key == ord('q'):  # press ESC or 'q' to exit
                    break
                elif key == ord('b'):  # press 'b' to capture the background
                    if not self._is_bg_captured:
                        self.createBackgroundModel(self._bg_sub_type, self._history, self._bg_sub_threshold, self._detect_shadows, self._learning_rate)
                        self._is_bg_captured = True
                        print('Background model created !')
                    else:
                        print('Background is already created !')
                elif key == ord('r'):  # press 'r' to reset the background
                    self.resetBackgroundModel()
                    self._is_bg_captured = False
                    print('Background model reset!')
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    hand = FingerDetection(0)
    hand.processVideo()


if __name__ == '__main__':
    main()

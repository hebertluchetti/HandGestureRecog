import cv2
import os
import copy
import sys
from src.utils.capture_config import CaptureConfig
from tensorflow.keras.models import load_model
from src.utils.training_config import TrainingConfig
from src.utils.file_utils import FileUtils
from src.utils.pos_process import PosProcess
from src.utils.background_subtraction import BackgroundSubtraction
from src.utils.hand_mask import HandMask
from logging.config import fileConfig
import logging


class BaseSegmentation:
    fileConfig('./logging_config.ini')
    logger = logging.getLogger()
    PNG_EXT = ".png"
    JPG_EXT = ".jpg"
    CLASS_MAP = TrainingConfig.CLASS_MAP_ITEMS
    UNKNOWN_CLASS = TrainingConfig.UNKNOWN_CLASS
    NUM_CLASSES = len(CLASS_MAP)
    CLASS_NAMES = list(CLASS_MAP.keys())
    START_MSG = "Press 's' to start/pause the capture"
    BACKGROUND_MSG = "Press 'b' and wait 5 secs to show the hand"
    RESET_BACKGROUND_MSG = "Press 'r' to reset the capture"
    QUIT_MSG = "Press 'Esc' or 'q' to quit"

    TRAINING_PATH = None
    TEST_PATH = None
    VALID_PATH = None
    CAPTURE_PATH = None

    WIDTH, HEIGHT = TrainingConfig.VGG16_WIDTH, TrainingConfig.VGG16_HEIGHT
    FILE_NAME_MODEL = TrainingConfig.FILE_NAME_MODEL
    FILE_WEIGHT_MODEL = TrainingConfig.FILE_WEIGHT_MODEL

    IMAGE_DATA_SET_PATH = TrainingConfig.IMAGE_DATA_SET_PATH
    IMAGE_TRAINING_SET_PATH = TrainingConfig.IMAGE_TRAINING_SET_PATH
    IMAGE_TEST_SET_PATH = TrainingConfig.IMAGE_TEST_SET_PATH
    IMAGE_VALID_SET_PATH = TrainingConfig.IMAGE_VALID_SET_PATH
    IMAGE_CAPTURE_PATH = TrainingConfig.IMAGE_CAPTURE_PATH
    IMG_SAVE_PATH = TrainingConfig.IMAGE_DATA_SET_PATH

    _bg_sub_type = BackgroundSubtraction.GAUSS_MIXTURE
    _bg_sub_threshold = BackgroundSubtraction.BACKGROUND_SUB_THRESHOLD  # background subtraction threshold
    _detect_shadows = BackgroundSubtraction.DETECT_SHADOWS
    _history = BackgroundSubtraction.HISTORY
    _learning_rate = BackgroundSubtraction.LEARNING_RATE  # 1.0 / history
    _cap = None
    _is_bg_captured = False  # bool, whether the background captured

    text_color = (0, 190, 255) #(0, 255, 255)
    red_color = (0, 0, 255)
    menu_color = (0, 80, 255)
    yellow_color = (0, 255, 255)
    green_color = (0, 255, 10)

    def __init__(self, source=0):
        self.logger.info('OpenCV: {}'.format(cv2.__version__))
        self.logger.info('Python: {}'.format(sys.version))
        self.logger.info('Version info: {}'.format(sys.version_info))
        self._init(source)

    def _init(self, source):
        self._file_utils = FileUtils()
        self._train_conf = TrainingConfig()
        self._cap_conf = CaptureConfig(source)
        self._pos_proc = PosProcess()
        self._bg_sub = BackgroundSubtraction()
        self._hand_mask = HandMask()
        self._dict_classes = None

    def createCaptureWin(self, win_name, width, height):
        # Define the capture window position on the screen
        x_cap_win, y_cap_win = int(width * 0.65), int(height * 0.012)
        # Create a named window and move it to (x_cap_win, y_cap_win)
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, x_cap_win, y_cap_win)

    def drawHandROI(self, frame, x_min, x_max, y_min, y_max, roi_color):
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), roi_color, 4)

    # def concat_tile(im_list_2d):
    #     return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    def showAllClasses(self, hand_class_dir, font):
        classes = []
        class_names = self.CLASS_NAMES

        for class_name in class_names:
            img, _ = self.loadImage(class_name, hand_class_dir)
            if img is not None:
                cv2.putText(img, class_name, (5, 15), font, 0.6, self.menu_color, 1, cv2.LINE_AA)
                classes.append(img)

        if len(classes) > 0:
            im_v = cv2.vconcat(classes)

            # Store the class images reference to use in predictions window
            self.getDictMenuImages(classes, hand_class_dir)

            cv2.imshow("Classes", im_v)

    def getDictMenuImages(self, classes, hand_class_dir):
        # Store the class images reference to use in predictions window
        img, _ = self.loadImage(self.UNKNOWN_CLASS, hand_class_dir)

        if img is not None:
            classes.append(img)

            img_class_names = list(self.CLASS_MAP.keys())
            img_class_names.append(self.UNKNOWN_CLASS)

            # Create a zip object from two lists
            zipbObj = zip(img_class_names, classes)
            # Create a dictionary from zip object
            self._dict_classes = dict(zipbObj)

    def loadImage(self, class_name, hand_class_dir, resize=True):
        file_img = "{}{}".format(class_name, self.JPG_EXT)
        img_path = os.path.join(hand_class_dir, file_img)

        if os.path.exists(img_path):
            self.logger.info("The image file exists in the directory:" + img_path)
        else:
            self.logger.info("The image file does not exist in the directory:" + img_path)
            return None

        original_img = cv2.imread(img_path)
        hand_class_img = self.resize(original_img, resize)

        return hand_class_img, original_img

    def resize(self, img, resize):
        if resize:
            img = cv2.resize(img, dsize=(0, 0), fx=0.55, fy=0.5)

        return img

    def loadModels(self):
        source_dir = TrainingConfig.MODEL_PATH
        model_path = self._file_utils.create_dirs(source_dir)
        load_path = os.path.join(model_path, self.FILE_NAME_MODEL)
        load_weight_path = os.path.join(model_path, self.FILE_WEIGHT_MODEL)

        self.logger.info("The model file will be loaded from the folder [{}]".format(load_path))
        self.logger.info("The weights file will be loaded from the folder [{}]".format(load_weight_path))

        if os.path.exists(load_path):
            self.logger.info("The model file exists in the directory:" + load_path)

            # Load the model
            model = load_model(load_path)
            weights = None

            if os.path.exists(load_weight_path):
                self.logger.info("The weight model file exists in the directory:" + load_weight_path)
                weights = model.load_weights(load_weight_path)
            else:
                self.logger.info("The weight model file not exists in the directory:" + load_weight_path)

            return model, weights
        else:
            self.logger.info("The model not exists in the directory:" + load_path)
        return None, None

    def getGestureName(self, val):
      return self.CLASS_NAMES[val]

    def joinPath(self, path, files):
        return os.path.join(path, files)

    def releaseAll(self, win_name):
        self._cap_conf.releaseCapture()
        self._cap = None
        cv2.destroyWindow(win_name)
        cv2.destroyAllWindows()

    def getHandImage(self, hand, frame, x_min, y_min, x_max, y_max, draw):
        # Background subtraction operation
        valid_hand = False
        hand_img = None
        result = True
        filtered_mask = None
        # Taking a copy of the image
        img = copy.deepcopy(frame)
        roi = img[y_min:y_max, x_min:x_max]  # clip the ROI

        # 1-) Create the _mask removing from the frame the background by using MOG/KNN segmentation algorithm
        #     and remove the noise from this _mask applying morphological filters
        fg_mask = self.removeBackground(roi, self._learning_rate)
        hand.setMask(fg_mask)

        if fg_mask is not None:
            # 2-) Find the hand _mask by selecting the biggest _contour and removing the forearm from it
            valid_hand, filtered_mask, max_contour, hand_img = self.findHandMask(img, roi, hand, draw)
            hand.setContour(max_contour)
            hand.setMask(filtered_mask)
            result = True

        return result, valid_hand, hand_img, filtered_mask

    # Create the _mask removing from the frame the background by using MOG/KNN segmentation algorithm
    # and remove the noise from this _mask applying morphological filters
    def createBackgroundModel(self, sub_type, history, bg_sub_threshold, detect_shadows, learning_rate):
        self._is_bg_captured = True
        self._bg_sub.createBackgroundModel(sub_type, history, bg_sub_threshold, detect_shadows)

    def resetBackgroundModel(self):
        self._is_bg_captured = False
        self._bg_sub.resetBackgroundModel()

    def removeBackground(self, img, learning_rate):
        # 1-) Create the _mask from the Gaussian Mixture-based Background/Foreground Segmentation
        # Apply GaussianBlur filter to the image and remove the background using MOG/KNN segmentation algorithm
        fg_mask = self._bg_sub.removeBackground(img, learning_rate)

        if fg_mask is not None:
            # 2-) Remove the noise from the _mask applying opening and closing morphological filter and GaussianBlur filter
            filtered_mask = self._pos_proc.removeMaskNoise(fg_mask)
            return filtered_mask

        return None

    def findHandMask(self, img, roi, hand, draw):
        consider_invalid_hand = False
        return self._hand_mask.findHandMask(img, roi, hand, consider_invalid_hand, draw)
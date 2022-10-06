import cv2
import os
import tkinter as tk
from logging.config import fileConfig
from src.utils.file_utils import FileUtils
import logging


class CaptureConfig:
    fileConfig('logging_config.ini')
    logger = logging.getLogger()

    def __init__(self, source=0):
        self._file_utils = FileUtils()
        self._source = source
        self._cap = None
        self.createVideoCapture()

    def createVideoCapture(self):
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        # aspect ratio of image aspect = w / h
        self._aspect_radio = self._width / self._height
        # Note that if aspect is greater than 1, then the image is oriented horizontally, while if it's less than 1,
        # the image is oriented vertically (and is square if aspect = 1).
        self.is_horizontal = self._aspect_radio > 1
        self.is_vertical = self._aspect_radio < 1
        self.is_square = self._aspect_radio == 1

        self.roi_color = (255, 255, 2)

        self.logger.info('Source capture camera: %s', self._source)
        self.logger.info('OpenCV resolution: {:.0f}x{:.0f} , Fps:{}'.format(self._width, self._height, self._fps))

    def getCapture(self):
        if self._cap is None:
            self.createVideoCapture()
        return self._cap

    def releaseCapture(self):
        self._cap.release()

    def make_1080p(self):
        self._cap.set(3, 1920)
        self._cap.set(4, 1080)

    def make_720p(self):
        self._cap.set(3, 1280)
        self._cap.set(4, 720)

    def make_480p(self):
        self._cap.set(3, 640)
        self._cap.set(4, 480)

    def change_res(self, width, height):
        self._cap.set(3, width)
        self._cap.set(4, height)

    def defineHandROI(self):
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.quit()

        # WIDTH, HEIGHT = 1536, 864
        x_min, y_min = 360, 50
        x_max, y_max = 630, 355
        area = (x_max - x_min) * (y_max - y_min)
        return (x_min, x_max, y_min, y_max, area, self.roi_color), width, height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

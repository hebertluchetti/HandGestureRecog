
from src.utils.contour import Contour

class Hand:
    def __init__(self):
        self._x_min, self._y_min = None, None
        self._x_max, self._y_max = None, None
        self._area = None
        self._contour = None
        self._h_contour = None
        self._mask = None
        self._centroid = None
        self._contour_proc = Contour()

    def setDimensions(self, x_min, y_min, x_max, y_max):
        self._x_min, self._y_min = x_min, y_min
        self._x_max, self._y_max = x_max, y_max
        self._area = (x_max - x_min) * (y_max - y_min)

    def getDimensions(self):
        return self._x_min, self. _y_min, self._x_max, self._y_max, self._area

    def setContour(self, contour):
        self._contour = contour

        if contour is not None:
            self._h_contour = self._contour_proc.contourToList(contour)
            cx, cy = self._contour_proc.centroid(contour)
            self._centroid = tuple([cx, cy])
        else:
            self._h_contour = None
            self._centroid = None

    def getContour(self):
        return self._contour

    def setHContour(self, h_contour):
        self._h_contour = h_contour

    def getHContour(self):
        return self._h_contour

    def setMask(self, mask):
        self._mask = mask

    def getMask(self):
        return self._mask

    def getCentroid(self):
        return self._centroid

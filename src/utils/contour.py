import cv2
import copy
import numpy as np
import math

from src.utils.contour_feature import ContourFeature

class Contour:
    def __init__(self):
        self._cnt_feature = ContourFeature()

    def perimeter(self, contour):
        return self._cnt_feature.perimeter(contour)

    def approximation(self, contour, accuracy):
        # Draw the approx of max_contour
        return self._cnt_feature.approximation(contour, accuracy)

    def centroid(self, max_con):
        # Find the _contour center
        cx, cy = self._cnt_feature.contourCentroid(max_con)
        return cx, cy

    def contourToList(self, contour):
        # Contour is a list of lists of points, cat them and squeeze out extra dimensions
        # Stack arrays in sequence vertically (row wise)
        return np.vstack(contour).squeeze()

    def getBiggestContour(self, contours):
        # Sort the contours from the biggest to the smallest and get the first
        return sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    def getContourFeatures(self, contour):
        return self._cnt_feature.momentFeatures(contour)

    def getMaxContour(self, contours):
        # max_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        # Find _contour of max _area
        max_contour = max(contours, key=cv2.contourArea)
        # Draw the approximation of biggest _contour
        approx = self.approximation(max_contour, 0.0005)
        area = cv2.contourArea(approx)
        return approx, area

    def findContourExtremes(self, contour):
        if contour is not None:
            # Determine the most extreme points along the _contour
            extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            extBot = tuple(contour[contour[:, :, 1].argmax()][0])

            centx = np.sqrt(((extRight[0] + extLeft[0]) ** 2) / 4)
            centy = np.sqrt(((extRight[1] + extLeft[1]) ** 2) / 4)
            return extLeft, extRight, extTop, extBot, centx, centy
        else:
            return None, None, None, None

    def convertPointSequenceToContour(self, point_contour):
        return np.array(point_contour).reshape((-1, 1, 2)).astype(np.int32)

    # Find contours and the biggest of them
    def findContours(self, mask):
        contours = None
        max_contour = None
        area = None

        if mask is not None:
            contours, _ = cv2.findContours(copy.deepcopy(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Find _contour of max _area and the approximation of biggest _contour
                max_contour, area = self.getMaxContour(contours)

        return contours, max_contour, area

    # Version 1 : Begin
    # Perform PCA analysis: Draw a rotated ellipse and their axis from the biggest _contour
    ##########################################################################################
    def drawEllipseAxes1(self, img, contour, center_circle, radius_offset):
        # Find the orientation of each shape
        self._cnt_feature.getContourAxisFromPCA(img, contour, center_circle, radius_offset)

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            # Fits ellipse to current _contour
            cv2.ellipse(img, ellipse, (0, 0, 255), 2, cv2.LINE_AA)
    # Version 1 : End
    ##########################################################################################

    # Version 2 : Begin
    # Mestrado: Draw a rotated ellipse and their axis from the biggest _contour
    ##########################################################################################
    def drawEllipseAxes2(self, img, contour, color=(0, 0, 255)):
        cx, cy, major_axis_len, minor_axis_len, orientation = self.getContourFeatures(contour)
        if cx is not None:
            axes = (major_axis_len, minor_axis_len)
            centroid = (cx, cy)

            # Adjust for images with top-left origin
            orientation = (2.0 * math.pi) - orientation
            # b = np.linalg.norm(far - start)
            # c = np.linalg.norm(start - end)
            # angle = np.arccos((np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b))

            ellipse = centroid, axes, np.degrees(orientation)
            cv2.ellipse(img, ellipse, color, 2, cv2.LINE_AA)
            self.drawAxes(img, centroid, axes, orientation, (255, 0, 0))

    def drawAxes(self, img, centroid, axes, orientation, color):
        (cx, cy) = centroid
        if cx is not None:
            (major_axis_len, minor_axis_len) = axes
            angle_perp = orientation - (math.pi/2.0) # Value used to increment and decrement the angle

            major_axis_x = int(cx + major_axis_len * math.cos(orientation))
            major_axis_y = int(cy - major_axis_len * math.sin(orientation))
            cv2.line(img, centroid, (major_axis_x, major_axis_y), color, 1, cv2.LINE_AA, 0)

            minor_axis_x = int(cx + minor_axis_len * math.cos(angle_perp))
            minor_axis_y = int(cy - minor_axis_len * math.sin(angle_perp))
            cv2.line(img, centroid, (minor_axis_x, minor_axis_y), color, 1, cv2.LINE_AA, 0)
    # Version 2 : End
    ##########################################################################################

    def drawHandContour(self, frame, contours, center_circle, radius_offset):
        detect_hand = np.zeros(frame.shape, dtype=np.uint8)
        approx = None

        # http://www.fossreview.com/2018/05/part-2-studying-digital-image-with-opencv-python.html
        if len(contours) > 0:
            # Find max_contour of max _area and the approximation of biggest _contour
            #max_contour = max(contours, key=cv2.contourArea)
            max_contour, _ = self.getMaxContour(contours)

            # Draw the approximation of biggest _contour
            cv2.drawContours(detect_hand, [max_contour], 0, (255, 0, 0), 2)

            # Straight Bounding Rectangle
            # Draw the bounding rectangle of hand max_contour
            # x, y, w, h = cv2.boundingRect(max_contour)
            # cv2.rectangle(detect_hand, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Rotated Rectangle
            # Draw the bounding of Rotated Rectangle from the hand max_contour
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(detect_hand, [box], 0, (0, 255, 0), 2)

            # Draw a rotate ellipse and their axis from the biggest _contour
            self.drawEllipseAxes1(detect_hand, max_contour, center_circle, radius_offset)
            #self.drawEllipseAxes2(detect_hand, max_contour)

            # Show the skin in the image along with the _mask
            cv2.imshow("detect", detect_hand)

        return approx



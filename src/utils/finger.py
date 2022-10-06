import cv2
import numpy as np
import copy
from src.utils.contour import Contour
from src.utils.contour_feature import ContourFeature
from src.utils.geometry_utils import GeometryUtils
from src.utils.hand import Hand


class Finger:
    # Colors for the view
    color_start = (204, 204, 0)
    color_end = (204, 0, 204)
    color_far = (255, 0, 0)
    color_far1 = (211, 84, 0)
    color_contour = (0, 255, 0)
    color_top = (0, 130, 255)  # Upper point of _contour
    color_fingers = (0, 255, 255)
    color_angle = (0, 255, 255)
    color_d = (0, 255, 255)
    color_start_far = (204, 204, 0)
    color_far_end = (204, 0, 204)
    color_start_end = (0, 255, 255)
    msg_number_of_fingers = 'Found fingers: '

    DIST_MAX_TOP_CENTER = 110
    LIMIT_ANGLE_SUP = 90

    def __init__(self):
        self._contour_proc = Contour()
        self._contour_feature = ContourFeature()
        self._geom_utils = GeometryUtils()


    def calculateFingerAngle(self, contour, defects, i):
        # It returns an array where each row contains these values
        # [ start point, end point, farthest point, approximate distance to farthest point ]
        s, e, f, d = defects[i, 0]
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]

        # Find the triangle associated with each convex defect to determine angle
        a = np.linalg.norm(far - end)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(start - end)
        # apply cosine rule here
        angle = np.arccos((np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b))
        degree = np.degrees(angle)
        degree = int(degree)

        #s = (a + b + c) / 2
        #ar = sqrt(s * (s - a) * (s - b) * (s - c))
        # distance between point and convex hull
        #d = (2 * ar) / a

        return degree, start, end, far, d, angle

    def checkNoFinger(self, img, start_list, fingers, top, centroid, color_fingers):
        # If no start_list (or end_list) points have been stored, it can be 0 fingers found or 1 finger found
        # ----------------------------------------------------------------------------------------------
        one_finger = True

        if len(start_list) == 0:
            # Find the euclidean distance between the points point_a and point_b = numpy.linalg.norm(point_a - point_b)
            min_dist = np.linalg.norm(top - centroid)

            # Verify the max distance from the _centroid to the upper point to consider a hand in fist
            if min_dist >= self.DIST_MAX_TOP_CENTER:
                fingers = fingers + 1
                one_finger = False
                cv2.putText(img, '{}'.format(fingers), tuple(top), 1, 1.7, color_fingers, 1, cv2.LINE_AA)
        return one_finger, fingers

    def checkEachFinger(self, img, start_list, end_list, fingers, color_fingers):
        # If start_list points have been stored, the number of fingers found will be counted
        size = len(start_list)

        for i in range(size):
            fingers = fingers + 1
            cv2.putText(img, '{}'.format(fingers), tuple(start_list[i]), 1, 1.7, color_fingers, 1, cv2.LINE_AA)

            if i == size - 1:
                fingers = fingers + 1
                cv2.putText(img, '{}'.format(fingers), tuple(end_list[i]), 1, 1.7, color_fingers, 1, cv2.LINE_AA)
        return fingers

    def countContourFingers(self, frame, contours, y_min, y_max, x_min, x_max, hand, draw):
        # Sort the contours from the biggest to the smallers and get the first
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Count the fingers of only the biggest _contour
        for max_contour in cnts:
            self.countFingers(frame, max_contour, y_min, y_max, x_min, x_max, hand, draw)
            break

    # Filters the starting points and end points that should be from a finger
    def filterFinger(self, roi, max_contour, defects, start_list, end_list):
        # Verify if there are convex hull defects
        if defects is not None:
            for i in range(defects.shape[0]):
                angle, start, end, far, d, _ = self.calculateFingerAngle(max_contour, defects, i)

                # Discard according to the distance between the start, end and most points inline, the angle and d
                #if (angle <= self.LIMIT_ANGLE_SUP and np.linalg.norm(start - end) > 20 and d > 12000):
                if angle <= self.LIMIT_ANGLE_SUP:  # Angle less than 90 degree, means a finger
                    # Collect all the starting and ending points that can represent fingers
                    start_list.append(start)
                    end_list.append(end)

                    # Visualization of representative points
                    cv2.circle(roi, tuple(start), 5, self.color_start, 2)
                    cv2.circle(roi, tuple(end), 5, self.color_end, 2)
                    cv2.circle(roi, tuple(far), 7, self.color_far, -1)
                # else:
                #     cv2.putText(roi, '{}'.format(angle),tuple(far), 1, 1.5, self.color_angle, 2, cv2.LINE_AA)
                #     cv2.putText(roi, '{}'.format(d),tuple(far), 1, 1.1, self.color_d, 1, cv2.LINE_AA)
                #     cv2.line(roi, tuple(start), tuple(far), self.color_start_far, 2)
                #     cv2.line(roi, tuple(far), tuple(end), self.color_far_end, 2)
                #     cv2.line(roi, tuple(start), tuple(end), self.color_start_end, 2)

    # Reference links:
    #   https://medium.com/@soffritti.pierfrancesco/handy-hands-detection-with-opencv-ac6e9fb3cec1
    #   https://omes-va.com/contando-dedos-defectos-de-convexidad-python-opencv/
    def countFingers(self, frame, win_name, hand, draw=False):
        x_min, y_min, x_max, y_max, _ = hand.getDimensions()
        roi = frame[y_min:y_max, x_min:x_max]
        fingers = 0  # Counter for the number of fingers raised
        defects = None
        max_contour = hand.getContour()

        # Draw only the biggest contour ust the first contour
        if max_contour is not None:
            # Find the contour center
            (cx, cy) = hand.getCentroid()
            center = tuple([cx, cy])

            # Upper point of the contour
            top = max_contour.min(axis=1)[0]

            # Found Contour by cv2.convexHull
            hull1 = cv2.convexHull(max_contour)

            # Convex Hull defects
            hull2 = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull2)

            # Verify if there are convex hull defects
            if defects is not None:
                start_list = []  # List where the initial points of convex defects are stored.
                end_list = []  # List where the end points of convex defects are stored.
                fingers = 0  # Counter for the number of fingers raised

                # Filters the starting points and end points that should be from a finger
                self.filterFinger(roi, max_contour, defects, start_list, end_list)

                # If no start_list (or end_list) points have been stored, it can be 0 fingers  or 1 finger found
                no_finger, fingers = self.checkNoFinger(roi, start_list, fingers, top, [cx, cy], self.color_fingers)

                # If start_list points have been stored, the number of fingers found will be counted
                fingers = self.checkEachFinger(roi, start_list, end_list, fingers, self.color_fingers)

        if draw:
            if max_contour is not None:
                # Draw the _contour center
                cv2.circle(roi, center, 5, (0, 255, 0), -1)
                # Draw the upper point of the _contour
                cv2.circle(roi, tuple(top), 5, self.color_top, -1)
                # Draw the _contour by cv2.convexHull
                cv2.drawContours(roi, [hull1], 0, self.color_contour, 2)

            if defects is not None:
                # The number of fingers found is displayed in the left rectangle
                cv2.putText(frame, '{}{}'.format(self.msg_number_of_fingers, fingers), (x_min, y_min - 5), 1, 1.8, self.color_fingers, 2, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

        return fingers

    def drawFingers(self, frame, max_contour, y_min, y_max, x_min, x_max, radius_offset):
        img = copy.deepcopy(frame)
        roi = img[y_min:y_max, x_min:x_max]

        # Draw only the biggest _contour ust the first _contour
        if max_contour is not None:

            # Find the _contour center
            cx, cy = self._contour_proc.centroid(max_contour)
            centroid = tuple([cx, cy])
            hull1 = cv2.convexHull(max_contour)
            # Upper point of the _contour
            y_upper = max_contour.min(axis=1)[0]

            cv2.drawContours(roi, [max_contour], 0, (0, 255, 0), 2)
            cv2.drawContours(roi, [hull1], 0, (0, 0, 255), 2)
            cv2.circle(roi, centroid, 5, (0, 255, 0), -1)
            cv2.circle(roi, tuple(y_upper), 5, self.color_top, -1)

            #  convexity defect
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):  # calculate the angle
                        # It returns an array where each row contains these values
                        # [ start point, end point, farthest point, approximate distance to farthest point ]
                        s, e, f, d = defects[i][0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # define _area of hull and _area of hand
                        # areahull = cv2.contourArea(hull1)
                        # areacnt = cv2.contourArea(max_contour)
                        # # find the percentage of _area not covered by hand in convex hull
                        # arearatio = ((areahull - areacnt) / areacnt) * 100

                        # Find the triangle associated with each convex defect to determine angle
                        # a = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        # b = sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        # c = sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        #
                        # s = (a + b + c) / 2
                        # ar = sqrt(s * (s - a) * (s - b) * (s - c))
                        #
                        # # distance between point and convex hull
                        # d = (2 * ar) / a
                        # # apply cosine rule here
                        # angle = acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                        #
                        # apply cosine rule here
                        # angle = acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        # ang = np.degrees(angle)
                        # ang = int(ang)
                        #cv2.putText(roi, '{}'.format(ang), tuple(far), 1, 1.2, self.color_angle, 1, cv2.LINE_AA)
                        #cv2.putText(roi, '{}'.format(d), tuple(far), 1, 0.9, self.color_d, 1, cv2.LINE_AA)

                        min_dist, _ = self._geom_utils.calculateDistanceAndAngle(far, centroid)

                        # Verify the max distance from the furthest point to the upper point to consider a hand in fist
                        if min_dist <= radius_offset:
                            cv2.circle(roi, far, 5, self.color_far, -1)
                            # cv2.line(roi, start, end, [0, 255, 0], 2)

            cv2.imshow('Fingers', roi)
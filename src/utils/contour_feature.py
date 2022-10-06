import cv2
import numpy as np
from math import pi, cos, sin, sqrt, atan2

from src.utils.geometry_utils import GeometryUtils


class ContourFeature:

    def __init__(self, contour=None):
        self._geom_utils = GeometryUtils()

    '''Funcao que retorna a _area de um contorno'''
    def area(self, contour):
        a = cv2.contourArea(contour)
        if a is None:
            a = 0.0
        return a

    '''Funcao que retorna o comprimento de um contorno'''
    def length(self, contour):
        l = cv2.arcLength(contour, True)
        if l is None:
            l = 0.0
        return l

    '''Funcao que retorna apenas os contornos com _area dentro de um range'''
    def contour_filter(self, area):
        a = cv2.contourArea(area)
        if a is None:
            a = 0.0
        return 300 <= a <= 2000

    '''Funcao que retorna a largura de um contorno'''
    def width(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        if w is None:
            w = 0.0
        return w

    '''Funcao que retorna a altura de um contorno'''
    def heigth(self,contour):
        x, y, w, h = cv2.boundingRect(contour)
        if h is None:
            h = 0.0
        return h

    '''Funcao que retorna a circularidade de um contorno'''
    def circularity(self, contour):
        c = (4 * pi * cv2.contourArea(contour)) / ((cv2.arcLength(contour, True) ** 2))
        if c is None:
            c = 0.0
        return c

    '''Funcao que retorna o alongamento de um contorno'''
    def elongation(self, m):
        x = m['mu20'] + m['mu02']
        y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
        e = (x + y ** 0.5) / (x - y ** 0.5)
        if e is None:
            e = 0.0
        return e

    def aspectRatio(self, contour):
        # It is the ratio of WIDTH to HEIGHT of bounding rect of the _contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return aspect_ratio

    def perimeter(self, contour):
        return cv2.arcLength(contour, True)

    def approximation(self, contour, accuracy):
        # Draw the approx of max_contour
        epsilon = accuracy * self.perimeter(contour)
        return cv2.approxPolyDP(contour, epsilon, True)

    def contourCentroid(self, contour):
        # Initialize the moment from _contour
        mo = cv2.moments(contour)
        # Find the _contour center
        return self.centroid(mo)

    def centroid(self, moment):
        # Find the _contour center
        if moment is not None and moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
        else:
            cx = None
            cy = None

        return cx, cy

    def getAllCentroids(self, image, contours, color=None, draw=True):
        centers = []
        # Calculate center-points
        for contour in contours:
            # Calculate mass-centers
            cx, cy = self.contourCentroid(contour)

            # Draw center in original image
            if draw:
                if color is None:
                    color = (255, 0, 0)
                    cv2.circle(image, (cx, cy), 1, color, 2)

            # Add center in list
            center = np.array([cx, cy])
            centers.append(center)

        return centers

    def getOrientation(self, moment):
        if moment["m00"] == 0:
            return None	# no region with the given value

        numerator = 2.0 * moment["mu11"]
        denominator = moment["mu20"] - moment["mu02"]

        # Calculate the orientation in radians
        if numerator == 0.0 or denominator == 0.0:
            orientation = 0.0
        else:
            orientation = atan2(numerator, denominator) / 2

        if orientation < 0:
            orientation = orientation + self._geom_utils.PI2 #6.283185307179586476925286766559 # orientation + 2*math.pi

        return orientation

    def momentFeatures(self, contour):
        # Initialize the moment from _contour
        mo = cv2.moments(contour)
        area = mo["m00"]

        if area == 0:
            return None

        mu_sum = mo["mu20"] + mo["mu02"]
        mu_diff = mo["mu20"] - mo["mu02"]

        val = np.power(mu_diff, 2) + 4 * np.power(mo["mu11"], 2)
        common = sqrt(val)

        # min_inertia e max_inertia are the minimum and maximum moments of inertia.
        # Together called the "principal moments of intertia"
        max_inertia = (mu_sum + common)/2
        min_inertia = (mu_sum - common)/2

        major_axis_len = self._geom_utils.SQRT2 * sqrt(min_inertia / area) # length
        minor_axis_len = self._geom_utils.SQRT2 * sqrt(max_inertia / area) # WIDTH

        if major_axis_len < minor_axis_len:
            major_axis_len, minor_axis_len = minor_axis_len, major_axis_len

        # spreadness = mu_sum / pow(_area, 2)
        # aspect_ratio = major_axis_len / minor_axis_len
        # eccentricity = val / pow(mu_sum, 2)
        #
        # # The normalized radius of gyration (R)
        # gyration_radius = math.sqrt(mu_sum / _area)

        #  Calculate the angle orientation in radian
        orientation = self.getOrientation(mo)

        # Find the _contour center
        cx, cy = self.centroid(mo)

        return cx, cy, major_axis_len, minor_axis_len, orientation

    def drawAxis(self, img, center, point, colour, scale, radius_offset):
        # http://man.hubwiz.com/docset/OpenCV.docset/Contents/Resources/Documents/d1/dee/tutorial_introduction_to_pca.html
        c = list(center)
        p = list(point)
        #https://stackoverflow.com/questions/53334694/get-perimeter-of-pixels-around-centre-pixel
        hypotenuse, angle = self._geom_utils.calculateDistanceAndAngle(c, p)

        if radius_offset is not None:
            # Here we lengthen the arrow by a factor of scale
            p[0] = c[0] - radius_offset * 2.0 * cos(angle)
            p[1] = c[1] - radius_offset * 2.0 * sin(angle)
        else:
            hypotenuse = scale * hypotenuse
            # Here we lengthen the arrow by a factor of scale
            p[0] = c[0] - scale * hypotenuse * cos(angle)
            p[1] = c[1] - scale * hypotenuse * sin(angle)

        # p[0] = c[0] - hypotenuse * math.cos(angle) + radius_offset
        # p[1] = c[1] - hypotenuse * math.sin(angle) + radius_offset
        cv2.line(img, (int(c[0]), int(c[1])), (int(p[0]), int(p[1])), colour, 1, cv2.LINE_AA)

        # create the arrow hooks
        c[0] = p[0] + 9 * cos(angle + pi / 4)
        c[1] = p[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(c[0]), int(c[1])), (int(p[0]), int(p[1])), colour, 1, cv2.LINE_AA)

        c[0] = p[0] + 9 * cos(angle - pi / 4)
        c[1] = p[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(c[0]), int(c[1])), (int(p[0]), int(p[1])), colour, 1, cv2.LINE_AA)

        return (int(p[0]), int(p[1])), angle

    def processContourPCA(self, contour):
        # [pca]
        # Construct a buffer used by the pca analysis
        sz = len(contour)
        data_pts = np.empty((sz, 2), dtype=np.float64)

        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = contour[i, 0, 0]
            data_pts[i, 1] = contour[i, 0, 1]

        # Perform PCA analysis
        mean = np.empty(0)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        # Store the center of the object
        center = (int(mean[0, 0]), int(mean[0, 1]))
        # [pca]

        factor = 0.02
        p1 = (center[0] + factor * eigenvectors[0, 0] * eigenvalues[0, 0], center[1] + factor * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (center[0] - factor * eigenvectors[1, 0] * eigenvalues[1, 0],  center[1] - factor * eigenvectors[1, 1] * eigenvalues[1, 0])

        return center, p1, p2

    def getContouBoundingRect (self, img, contour):
        # Straight Bounding Rectangle
        # Draw the bounding rectangle of hand max_contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return x + w, y + h

    def getContourAxisFromFitLine(self, contour, rows, cols):
        [vx, vy, cx, cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        lefty = int((-cx * vy / vx) + cy)
        righty = int(((cols - cx) * vy / vx) + cy)
        pt1 = (cols - 1, righty)
        pt2 = (0, lefty)

        return pt1, pt2, cx, cy

    def getContourAxisFromPCA(self, img, contour, center_circle, radius_offset):
        # Perform PCA analysis and store the center of the object
        center, p1, p2 = self.processContourPCA(contour)

        # Draw the axis components
        cv2.circle(img, center, 3, (255, 0, 255), 2)
        p_1, ang1 = self.drawAxis(img, center, p1, (0, 255, 0), 1, radius_offset)
        p_2, ang2 = self.drawAxis(img, center, p2, (255, 255, 0), 1, None)

        # Draw ellipse around the hand _contour
        # c = list(center)
        # c = np.asarray(c, dtype=np.intc)
        # p_1 = np.asarray(p_1, dtype=np.intc)
        # p_2 = np.asarray(p_2, dtype=np.intc)
        # minor_axis = np.linalg.norm(c - p_1)
        # major_axis = np.linalg.norm(c - p_2)
        #
        # if minor_axis > major_axis:
        #     major_axis, minor_axis = minor_axis, major_axis
        #     ang2 = ang1
        #
        # ang2 = np.degrees(ang2)
        # ang2 = int(ang2)
        # ellipse = center, (major_axis*2.0, minor_axis*2.0), ang2

        # Fits ellipse to current _contour
        #cv2.ellipse(img, ellipse, (0, 0, 255), 2, cv2.LINE_AA)

        return p_1, p_2, ang1, ang2



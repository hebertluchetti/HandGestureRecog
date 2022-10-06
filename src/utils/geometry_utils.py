#from math import sqrt, fabs, atan2, pi
#import numpy as np
from numpy import arccos, array, dot, pi, cross, sqrt, arctan2, power, fabs, mean, argsort, apply_along_axis, arctan
from numpy.linalg import det, norm
from scipy.spatial import distance as dist


class GeometryUtils:
    def __init__(self):
        sqrt_2 = sqrt(2.0)
        two_pi = 2.0 * pi


    def sortPoints(self, horiz_contour):
        # Find the Center of Mass: data is a numpy array of shape (Npoints, 2)
        mean1 = mean(horiz_contour, axis=0)
        # Compute angles
        angles = arctan2((horiz_contour - mean1)[:, 1], (horiz_contour - mean1)[:, 0])
        # Transform angles from [-pi,pi] -> [0, 2*pi]
        angles[angles < 0] = angles[angles < 0] + 2 * pi
        # Sort
        sorting_indices = argsort(angles)

        # for ordering points(transform, adjust for 0 to 2pi, argsort, index at points)
        sorted_data = []
        for idx in sorting_indices:
            sorted_data.append(horiz_contour[idx])

        return sorted_data

    def orderPoints(self, horiz_contour):
        pts = box = array(horiz_contour)
        # sort the points based on their x-coordinates
        # Using sorted and lambda
        pts_l = list(horiz_contour)
        xSorted = sorted(pts_l, key=lambda pt: pt[0])

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2]
        rightMost = xSorted[-2:]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = sorted(leftMost, key=lambda pt: pt[1])#leftMost[argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        dists = []
        for i in range(len(rightMost)):
            pt = rightMost[i]
            x = pt[0]
            y = pt[1]
            # Find the distance from the center-line-intersection point to the _contour-offset intersections
            dist, _ = self.calculateDistanceAndAngle(tl, (x, y))
            dists.append(dist)

        if len(dists) > 1:
            zipped = zip(rightMost, dists)
            z = list(zipped)

            # Using sorted and lambda
            res = sorted(z, key=lambda item: item[1], reverse=True)
            print(res)
            (br, tr) = res[:2]
            br = br[0]
            tr = tr[0]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return array([tl, tr, br, bl], dtype="float32")

    def dimensionInfo (self, width, height):
        # aspect ratio of image aspect = w / h
        aspect_radio = width / height
        # Note that if aspect is greater than 1, then the image is oriented horizontally, while if it's less than 1,
        # the image is oriented vertically (and is square if aspect = 1).
        is_horizontal = aspect_radio > 1
        is_vertical = aspect_radio < 1
        is_square = aspect_radio == 1
        return aspect_radio, is_horizontal, is_vertical, is_square

    # from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
    def point_line_distance(self, pta, ptb, pt):
        """ segment line AB, point P, where each one is an array([x, y]) """
        if all(pta == pt) or all(ptb == pt):
            return 0
        if arccos(dot((pt - pta) / norm(pt - pta), (ptb - pta) / norm(ptb - pta))) > pi / 2:
            return norm(pt - pta)
        if arccos(dot((pt - ptb) / norm(pt - ptb), (pta - ptb) / norm(pta - ptb))) > pi / 2:
            return norm(pt - ptb)
        return norm(cross(pta - ptb, pta - pt)) / norm(ptb - pta)

    """ 
        Function to find distance from a point to a line
        Now you have to find A, B, C, x, and y.
        
        x = (x ,y)
        y = (x, y)
        coef = np.polyfit(x, y, 1)
        A = coef[0]
        B = coef[1]
        C = A*x[0] + B*x[1]
        ================================
        
        norm = np.linalg.norm

        p1 = np.array([0,-4/3])
        p2 = np.array([2, 0])
        
        p3 = np.array([5, 6])
        d = np.abs(norm(np.cross(p2-p1, p1-p3)))/norm(p2-p1)
            """
    def point_line_dist_coef(self, x1, y1, a, b, c):
        d = abs((a * x1 + b * y1 + c)) / (sqrt(a * a + b * b))
        return d

    # Mid point of a line
    def midPoint(self, pt1, pt2):
        return [(pt1[0] + pt2[0]) * 0.5, (pt1[1] + pt2[1]) * 0.5]

    def calculateDistanceAndAngle(self, p1, p2):
        angle, dist = None, None

        if p1 is not None and p2 is not None:
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            x_diff = x1 - x2
            y_diff = y1 - y2

            # angle in radians
            angle = arctan2(y_diff, x_diff)
            # Euclidean Distance
            hypotenuse = sqrt(power(x_diff, 2) + power(y_diff, 2))

            # y_diff = p[1] - q[1]
            # x_diff = p[0] - q[0]
            # angle = atan2(y_diff, x_diff) # angle in radians
            # hypotenuse = sqrt(y_diff * y_diff + x_diff * x_diff)
            # hypotenuse = sqrt(y_diff * y_diff + x_diff * x_diff)

        return hypotenuse, angle

    # Detect the intersection between the line and a circle with the center
    # equal to the determined center by the distance transform from the hand _mask
    # and the radius equal to the maximum distance from that center to the border
    # with the palm hand orientation
    def findCircleLineIntersections(self, circle_center, circle_radius, pt1, pt2):
        intersections = []
        has_inters, point = self.findCircleLineIntersection(circle_center, circle_radius, pt1, pt2)

        if has_inters and point:
            intersections = intersections + point

        return intersections

    def findCircleLineIntersection(self, center, radius, pt1, pt2):
        intersections = []
        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, center
        dx, dy = p2x - p1x, p2y - p1y
        a = dx * dx + dy * dy
        b = 2.0 * (dx * (p1x - cx) + dy * (p1y - cy))

        cc = (cx * cx + cy * cy) + (p1x * p1x + p1y * p1y) - \
             (2 * (cx * p1x + cy * p1y)) - radius * radius

        deter = (b * b) - (4.0 * a * cc)

        # if (i < 0), then we are totally outside the circle
        if deter < 0:
            result = False  # "Outside"
        else:
            # if (i == 0), then this line is tangent to the circle,
            if deter == 0:
                result = True  # "Tangent" NOTE: should calculate this point
            else:
                e = sqrt(deter)
                d = 2.0 * a
                u1 = (-b + e) / d
                u2 = (-b - e) / d

                if (u1 < 0 or u1 > 1) and (u2 < 0 or u2 > 1):
                    if (u1 < 0 and u2 < 0) or (u1 > 1 and u2 > 1):
                        result = False  # "Outside"
                    else:
                        result = False  # "Inside"
                else:
                    result = True  # "Intersection"

                if 0 <= u1 <= 1:
                    inters1_x = p1x + u1 * dx
                    inters1_y = p1y + u1 * dy
                    intersections.append((inters1_x, inters1_y))

                if 0 <= u2 <= 1:
                    inters2_x = p1x + u2 * dx
                    inters2_y = p1y + u2 * dy
                    intersections.append((inters2_x, inters2_y))

        return result, intersections

    # https://stackoverflow.com/questions/30844482/what-is-most-efficie
    def findCircleLineIntersection_1(self, circle_center, circle_radius, pt1, pt2, full_line=True,
                                     tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.
            False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider
            it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which
            the circle intercepts a line segment.
        """
        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2) ** .5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
        intersections = []
        inters = []

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
                 cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
                for sign in
                ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
            if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
                fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                          intersections]
                intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) == 2 and abs(
                    discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)

                inters.append(list(intersections[0]))
                return inters
            else:
                for point in intersections:
                    inters.append(list(point))
                return inters

    # Detect the intersection between the line and a hand _contour
    def findLineAndContourIntersections(self, img, horiz_contour, pta, ptb):
        intersections = []
        # horiz_contour - Contour is a list of lists of points, cat them and squeeze out extra dimensions
        size, _ = horiz_contour.shape

        for i in range(size):
            pt1 = horiz_contour[i]
            index = i + 1

            if index == size:
                index = 0

            pt2 = horiz_contour[index]

            # key = cv2.waitKey()
            # img2 = np.ones(img.shape, dtype=np.uint8)
            # cv2.line(img2, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
            # cv2.line(img2, tuple(pta), tuple(ptb), (0, 0, 255), 2)
            #print("pt1=", pt1, ", pt2=", pt2)
            # cv2.imshow('Test', img2)

            has_inters, point = self.findLineLineIntersection(pta, ptb, pt1, pt2)

            if has_inters:
                #print("point=", point)
                intersections += point

        # print("Fim =", intersections)
        # cv2.imshow('Test', img2)
        # key = cv2.waitKey()
        return intersections

    # Find the perpendicular line to a line
    def findPerpLineCoords(self, aX, aY, bX, bY, length):
        vX = bX - aX
        vY = bY - aY

        if vX == 0 or vY == 0:
            return 0, 0, 0, 0

        mag = sqrt(vX * vX + vY * vY)
        vX = vX / mag
        vY = vY / mag
        vX, vY = 0 - vY, vX

        cX = bX + vX * length
        cY = bY + vY * length
        dX = bX - vX * length
        dY = bY - vY * length

        #return int(cX), int(cY), int(dX), int(dY)
        return cX, cY, dX, dY

    # http://pythonshort.blogspot.com/2015/03/intersection-of-two-line-segments-in.html
    # https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    # This gives the point of intersection of two lines, but if we are given line segments instead of lines, we have to also recheck that the point so computed actually lies on both the line segments.
    # If the line segment is specified by points (x1, y1) and (x2, y2), then to check if (x, y) is on the segment we have to just check that
    #
    # min (x1, x2) <= x <= max (x1, x2)
    # min (y1, y2) <= y <= max (y1, y2)
    def hasIntersection(self, pa1, pa2, pb1, pb2):
        ax, ay, bx, by, cx, cy, dx, dy = pa1[0], pa1[1], pa2[0], pa2[1], pb1[0], pb1[1], pb2[0], pb2[1]
        left = max(min(ax, bx), min(cx, dx))
        right = min(max(ax, bx), max(cx, dx))
        top = max(min(ay, by), min(cy, dy))
        bottom = min(max(ay, by), max(cy, dy))

        if top > bottom or left > right:
            return False, ('NO INTERSECTION', list())
        if (top, left) == (bottom, right):
            return True, ('POINT INTERSECTION', list((left, top)))
        return True, ('SEGMENT INTERSECTION', list((left, bottom, right, top)))

    def determinant(self, a, b):
        return a[0] * b[1] - a[1] * b[0]

    def findLineLineIntersection(self, pa1, pa2, pb1, pb2):
        intersection = []
        DET_TOLERANCE = 0.00000001
        ret, item = self.hasIntersection(pa1, pa2, pb1, pb2)

        if not ret:
            return False, intersection

        # Line AB represented as a1x + b1y = c1
        ax, ay, bx, by = pa1[0], pa1[1], pa2[0], pa2[1]
        a1 = by - ay
        b1 = ax - bx
        c1 = a1 * ax + b1 * ay

        # Line CD represented as a2x + b2y = c2pc
        cx, cy, dx, dy = pb1[0], pb1[1], pb2[0], pb2[1]
        a2 = dy - cy
        b2 = cx - dx
        c2 = a2 * cx + b2 * cy

        determinant = a1 * b2 - a2 * b1

        #if fabs(determinant) < DET_TOLERANCE or determinant == 0:
        if determinant == 0:
            # if self.almostEqual(determinant, 0.0):
            # The lines are parallel.This is simplified by returning a pair None
            result = False
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            # This gives the point of intersection of two lines, but if we are given line segments instead of lines, we have to also recheck that the point so computed actually lies on both the line segments.
            # If the line segment is specified by points (x1, y1) and (x2, y2), then to check if (x, y) is on the segment we have to just check that
            #
            # min (x1, x2) <= x <= max (x1, x2)
            # min (y1, y2) <= y <= max (y1, y2)
            intersection.append([int(x), int(y)])
            result = True

        return result, intersection

    def findLineLineIntersection_1(self, pa1, pa2, pb1, pb2):
        intersection = []
        DET_TOLERANCE = 0.00000001

        # Teste 1 ======================================================================
        #http://www-cs.ccny.cuny.edu/~wolberg/capstone/intersection/Intersection%20point%20of%20two%20lines.html
        s10_x = pa2[0] - pa1[0] # p1.x - p0.x
        s10_y = pa2[1] - pa1[1] # p1.y - p0.y
        s32_x = pb2[0] - pb1[0] # p3.x - p2.x
        s32_y = pb2[1] - pb1[1] # p3.y - p2.y
        denom = s10_x * s32_y - s32_x * s10_y

        if denom < 0:
            return False, intersection

        s02_x = pa1[0] - pb1[0] # p0.x - p2.x
        s02_y = pa1[1] - pb1[1] # p0.y - p2.y

        s_numer = s10_x * s02_y - s10_y * s02_x
        if s_numer < 0:
            return False, intersection

        t_numer = s32_x * s02_y - s32_y * s02_x
        if t_numer < 0:
            return False, intersection

        if s_numer > denom or t_numer > denom:
            return False, intersection

        t = t_numer / denom
        x = pa1[0] + (t * s10_x)
        y = pa1[1] + (t * s10_y)
        intersection.append([int(x), int(y)])

        return True, intersection

    def findLineLineIntersection_2(self, pa1, pa2, pb1, pb2):
        intersection = []
        DET_TOLERANCE = 0.00000001

        # Teste 2 ======================================================================
        #Line equationsax + by = c - (Pa1Pa2)(Pb1Pb2)
        a1x, a1y, a2x, a2y = pa1[0], pa1[1], pa2[0], pa2[1]
        a1 = a1y - a2y
        b1 = a2x - a1x
        c1 = a1 * a1x + b1 * a1y

        b1x, b1y, b2x, b2y = pb1[0], pb1[1], pb2[0], pb2[1]
        a2 = b1y - b2y
        b2 = b2x - b1x
        c2 = a2 * b1x + b2 * b1y

        # Line intersections
        detX = (a1 * b2) - (a2 * b1)
        detY = (a2 * b1) - (a1 * b2)

        # Parallel lines
        if detX == 0 or detY == 0:
            return False, intersection

        xi = (c1 * b2 - b1 * c2) / detX
        yi = (a2 * c1 - a1 * c2) / detY
        intersection.append([int(xi), int(yi)])

        return True, intersection

    # https://www.cs.hmc.edu/ACM/lectures/intersections.html
    def findLineLineIntersection_3(self, pt1, pt2, ptA, ptB):
        """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
            returns a tuple: (xi, yi, valid, r, s), where
            (xi, yi) is the intersection
            r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
            s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
                valid == 0 if there are 0 or inf. intersections (invalid)
                valid == 1 if it has a unique intersection ON the segment    """

        DET_TOLERANCE = 0.00000001
        intersection = []

        # the first line is pt1 + r*(pt2-pt1)
        # in component form:
        x1, y1 = pt1
        x2, y2 = pt2
        dx1 = x2 - x1
        dy1 = y2 - y1

        # the second line is ptA + s*(ptB-ptA)
        x, y = ptA
        xB, yB = ptB
        dx = xB - x
        dy = yB - y

        """ We need to find the (typically unique) values of r and s that will satisfy
            (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
            which is the same as
               [ dx1  -dx ][ r ] = [ x-x1 ]
               [ dy1  -dy ][ s ] = [ y-y1 ]
            whose solution is
               [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
               [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    
            where DET = (-dx1 * dy + dy1 * dx), if DET is too small, they're parallel """

        DET = (-dx1 * dy + dy1 * dx)

        if fabs(DET) < DET_TOLERANCE:
            return False, intersection

        # now, the determinant should be OK
        DETinv = 1.0 / DET

        # find the scalar amount along the "self" segment
        r = DETinv * (-dy * (x - x1) + dx * (y - y1))
        # find the scalar amount along the input line
        s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

        # return the average of the two descriptions
        xi = (x1 + r * dx1 + x + s * dx) / 2.0
        yi = (y1 + r * dy1 + y + s * dy) / 2.0

        intersection.append([int(xi), int(yi)])
        return True, intersection

    def almostEqual(self, number1, number2):
        EPSILON = 1e-5
        return abs(number1 - number2) <= (EPSILON * max(1.0, abs(number1), abs(number2)))

    def polygonCentroid1(self, arr):
        length = arr.shape[0]
        sum_x = sum(arr[:, 0])
        sum_y = sum(arr[:, 1])
        return sum_x / length, sum_y / length

    def polygonCentroid3(self, data):
        x, y = zip(*data)
        l = len(x)
        return sum(x) / l, sum(y) / l

    def polygonCentroid3(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = sum(_x_list) / _len
        _y = sum(_y_list) / _len
        return (_x, _y)

    #==================================================================================================
    #    Geometry functions to find intersecting lines.
    #    Thes calc's use this formula for a straight line:-
    #        y = mx + b where m is the gradient and b is the y value when x=0
    #
    #    See here for background http://www.mathopenref.com/coordintersection.html
    #
    #    Throughout the code the variable p is a point tuple representing (x,y)
    #    link: http://www.pygame.org/wiki/IntersectingLineDetection

    # Calc the gradient 'm' of a line between p1 and p2
    def calculateGradient(self, p1, p2):
        # Ensure that the line is not vertical
        if (p1[0] != p2[0]):
            m = (p1[1] - p2[1]) / (p1[0] - p2[0])
            return m
        else:
            return None

    # Calc the point 'b' where line crosses the Y axis
    def calculateYAxisIntersect(self, p, m):
        return p[1] - (m * p[0])

    # Calc the point where two infinitely long lines (p1 to p2 and p3 to p4) intersect.
    # Handle parallel lines and vertical lines (the later has infinate 'm').
    # Returns a point tuple of points like this ((x,y),...)  or None
    # In non parallel cases the tuple will contain just one point.
    # For parallel lines that lay on top of one another the tuple will contain
    # all four points of the two lines
    def getIntersectPoint(self, p1, p2, p3, p4):
        m1 = self.calculateGradient(p1, p2)
        m2 = self.calculateGradient(p3, p4)

        # See if the the lines are parallel
        if (m1 != m2):
            # Not parallel

            # See if either line is vertical
            if (m1 is not None and m2 is not None):
                # Neither line vertical
                b1 = self.calculateYAxisIntersect(p1, m1)
                b2 = self.calculateYAxisIntersect(p3, m2)
                x = (b2 - b1) / (m1 - m2)
                y = (m1 * x) + b1
            else:
                # Line 1 is vertical so use line 2's values
                if m1 is None and m2 is not None:
                    b2 = self.calculateYAxisIntersect(p3, m2)
                    x = p1[0]
                    y = (m2 * x) + b2
                # Line 2 is vertical so use line 1's values
                elif m2 is None and m1 is not None:
                    b1 = self.calculateYAxisIntersect(p1, m1)
                    x = p3[0]
                    y = (m1 * x) + b1
                else:
                    assert False

            return ((x, y),)
        else:
            # Parallel lines with same 'b' value must be the same line so they intersect
            # everywhere in this case we return the start and end points of both lines
            # the calculateIntersectPoint method will sort out which of these points
            # lays on both line segments
            b1, b2 = None, None  # vertical lines have no b value
            if m1 is not None:
                b1 = self.calculateYAxisIntersect(p1, m1)

            if m2 is not None:
                b2 = self.calculateYAxisIntersect(p3, m2)

            # If these parallel lines lay on one another
            if b1 == b2:
                return p1, p2, p3, p4
            else:
                return None

    def ellipseCenter(self, a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        num = b * b - a * c
        x0 = (c * d - b * f) / num
        y0 = (a * f - b * d) / num
        return array([x0, y0])

    def ellipseAngleOfRotation(self, a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        return 0.5 * arctan(2 * b / (a - c))

    def ellipseAngleOfRotation2(self, a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        if b == 0:
            if a > c:
                return 0
            else:
                return pi / 2
        else:
            if a > c:
                return arctan(2 * b / (a - c)) / 2
            else:
                return pi / 2 + arctan(2 * b / (a - c)) / 2

    def ellipseAxisLength(self, a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = sqrt(up / down1)
        res2 = sqrt(up / down2)
        return array([res1, res2])

    def findCircleTangent(self, circle_center, circle_radius, pt):
        Px = pt[0]
        Py = pt[1]
        Cx = circle_center[0]
        Cy = circle_center[1]
        r = circle_radius

        T1x, T1y, T2x, T2y = None, None, None, None
        dx, dy = Px - Cx, Py - Cy
        dxr, dyr = -dy, dx
        d = sqrt(dx ** 2 + dy ** 2)

        if d >= r:
            rho = r / d
            ad = rho ** 2
            bd = rho * sqrt(1 - rho ** 2)
            T1x = Cx + ad * dx + bd * dxr
            T1y = Cy + ad * dy + bd * dyr
            T2x = Cx + ad * dx - bd * dxr
            T2y = Cy + ad * dy - bd * dyr

            print('The tangent points:')
            print('\tT1≡(%g,%g),  T2≡(%g,%g).' % (T1x, T1y, T2x, T2y))
            if (d / r - 1) < 1E-8:
                print('P is on the circumference')
            else:
                print('The equations of the lines P-T1 and P-T2:')
                print('\t%+g·y%+g·x%+g = 0' % (T1x - Px, Py - T1y, T1y * Px - T1x * Py))
                print('\t%+g·y%+g·x%+g = 0' % (T2x - Px, Py - T2y, T2y * Px - T2x * Py))
        else:
            print('''\
           Point P≡(%g,%g) is inside the circle with centre C≡(%g,%g) and radius r=%g.
           No tangent is possible...''' % (Px, Py, Cx, Cy, r))

        if T1x is not None:
            t1 = (T1x, T1y)
        else:
            t1 = None

        if T2x is not None:
            t2 = (T2x, T2y)
        else:
            t2 = None

        return t1, t2

    def get_square(self, image, out_size=None):
        w, h = image.size
        sz = min(w, h)
        if not out_size or out_size > sz:
            out_size = sz
        x_offset = (w - sz) // 2
        y_offset = (h - sz) // 2
        image = image.crop((x_offset, y_offset, x_offset + sz, y_offset + sz))
        image.thumbnail((out_size, out_size))
        return image
import cv2
from math import ceil
from numpy import zeros

from src.utils.contour import Contour
from src.utils.contour_feature import ContourFeature
from src.utils.geometry_utils import GeometryUtils
from src.utils.finger import Finger


class Forearm:
    def __init__(self):
        self._contour_proc = Contour()
        self._contour_feature = ContourFeature()
        self._geom_utils = GeometryUtils()
        self._finger_proc = Finger()
        self._offset_radius = 1.65
        self._percent_radius = 0.55
        self._dist_max = 80
        self._factor_dist_centers = 0.1042

    def replaceCenter(self, img, center_mass, center_df, offset, p_top, p_botton, height, consider_invalid_hand, draw):
        if consider_invalid_hand:
            valid_hand = True
        else:
            valid_hand = False

        is_moved_center = False
        new_radius = None
        new_center = None
        dist_centers, _ = self._geom_utils.calculateDistanceAndAngle(center_mass, center_df)
        dist_top_center, _ = self._geom_utils.calculateDistanceAndAngle(center_mass, p_top)
        dist_bot_top, _ = self._geom_utils.calculateDistanceAndAngle(p_botton, p_top)
        max_dist = height * self._factor_dist_centers

        # Determine the Y center of DF is higher the Y center of _centroid
        # In this case the center must be moved
        if center_df[1] > center_mass[1] and dist_centers > max_dist:
            # When the center is on a wrong place we can discard or not this frame
            # Depend pn the this flag consider_invalid_hand
            if consider_invalid_hand:
                is_moved_center = True
                diff_x = center_mass[0] - p_top[0]
                mid_x = abs(ceil(diff_x / 1.20))
                if diff_x < 0:
                    new_x = ceil(center_mass[0] + mid_x)
                else:
                    new_x = ceil(center_mass[0] - mid_x)

                if dist_bot_top > height * 0.84:# Open hand
                    new_radius = ceil(offset * 1.1)
                    new_center = [new_x, ceil(p_top[1] + dist_top_center/2.0)]
                    label = "Open"
                    color = (0, 255, 0)
                else: # Closed hand
                    new_radius = ceil(offset * 0.95)
                    new_center = [new_x, ceil(p_top[1] + dist_top_center/5.0)]
                    label = "Closed"
                    color = (0, 255, 255)

                if draw:
                    cv2.line(img, tuple(p_top), tuple(center_mass), (255, 0, 0), 2)
                    cv2.putText(img, '{}'.format(label), tuple(new_center), 1, 1.2, color, 1, cv2.LINE_AA)
                    # Draw the circle offset from _centroid
                    cv2.circle(img, tuple(new_center), int(new_radius), color, 2)
        else:
            valid_hand = True

        #print("replaceCenter valid_hand=", valid_hand )

        if is_moved_center:
            return valid_hand, is_moved_center, (ceil(new_center[0]), ceil(new_center[1])), new_radius
        else:
            if draw:
                cv2.putText(img, '{}'.format("DF center"), tuple(center_df), 1, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
                # Draw the circle offset from distTransform
                cv2.circle(img, center_df, int(offset), (255, 0, 0), 2)

            return valid_hand, is_moved_center, center_df, offset

    def removeForearmFromHand(self, img, width, height, hand, consider_invalid_hand, draw=False):
        if consider_invalid_hand:
            valid_hand = True
        else:
            valid_hand = False

        contour = hand.getContour()
        bg_mask = hand.getMask()

        # Process _contour of max _area and the the approximation of biggest _contour
        if contour is not None:
            # 1-) Determine the most extreme points along the _contour
            ext_left, ext_right, ext_top, ext_bot, center_x, center_y = self._contour_proc.findContourExtremes(contour)
            black_img = zeros((height, width, 3), dtype="uint8")

            # 2-) Draw a circle around the palm hand from distance transform
            circle_center, circle_radius, circle_offset = self.distTransform(bg_mask, draw)

            # Draw the contour before the forearm removal with the extreme points
            self.drawBeforeRemoval(black_img, contour, ext_left, ext_right, ext_top, ext_bot, center_x, center_y, draw)

            if circle_center is not None:
                df_center = circle_center
                # Calculate the mass center of _contour (by moment)
                (cx, cy) = hand.getCentroid()

                # 3-) Verify if the center was found correctly. If not it is needed to move it to a acceptable place
                valid_hand, is_moved_center, new_center, new_radius = self.replaceCenter(img, [cx, cy], circle_center, circle_offset, ext_top, ext_bot, height, consider_invalid_hand, draw)

                if is_moved_center and valid_hand:
                    circle_center, circle_offset = new_center, new_radius

                # 4-) Find the points that represent axis line from _contour
                #pt1, pt2, ang1, ang2 = self._contour_feature.getContourAxisFromPCA(black_img, max_contour, circle_center, circle_offset)
                pt1, pt2, clx, cly = self._contour_feature.getContourAxisFromFitLine(contour, height, width)

                # Determine the point with button Y value
                if pt1[1] > pt2[1]:
                    pt_y_max = pt1
                else:
                    pt_y_max = pt2

                # 5-) Find the intersection between the palm circle and a line from the circle_center with the palm orientation
                axis_circle_intersections = self._geom_utils.findCircleLineIntersections(circle_center, circle_offset, circle_center, pt_y_max)
                found, limits = self.findForearmLimits(img, contour, circle_center, circle_offset, axis_circle_intersections, draw)

                if found:
                    # 6-) Filter the _contour points from the forearm intersections with a circle
                    filtered_contour = self.filterForearmContour(contour, limits, cx)

                    if len(filtered_contour) > 2: # and cv2.isContourConvex(filtered_contour):
                        contour = filtered_contour

                # Draw the _contour centres the hand with forearm removal
                self.drawAfterRemoval(img, black_img, contour, circle_center, circle_offset, (cx, cy), df_center,
                                      pt_y_max, draw)

                return valid_hand, circle_center, circle_radius, circle_offset, contour

        return valid_hand, None, None, None, contour

    def drawBeforeRemoval(self, black_img, contour, ext_left, ext_right, ext_top, ext_bot, center_x, center_y, draw):
        # Draw the contour before the forearm removal with the extreme points
        if draw:
            # Draw hand _contour
            cv2.drawContours(black_img, [contour], 0, (0, 255, 0), 2)
            # Draw circles on extreme points
            cv2.circle(black_img, ext_left, 5, (0, 0, 255), -1)
            cv2.circle(black_img, ext_right, 5, (0, 255, 0), -1)
            cv2.circle(black_img, ext_top, 5, (255, 0, 0), -1)
            cv2.circle(black_img, ext_bot, 5, (255, 255, 0), -1)
            cv2.circle(black_img, (ceil(center_x), ceil(center_y)), 5, (255, 255, 255), -1)
            cv2.putText(black_img, 'Extreme Center', (ceil(center_x), ceil(center_y)), 1, 1.1, (255, 255, 255), 1, cv2.LINE_AA)

    def drawAfterRemoval(self, img, black_img, contour, circle_center, circle_offset, centroid, df_center, pt_y_max, draw):
        # Draw the contour centres the hand with forearm removal
        if draw:
            # Draw the line that represents the df line from contour
            cv2.line(black_img, df_center, pt_y_max, (0, 255, 255), 2)

            # Draw the mass center of contour (moment)
            cv2.circle(black_img, centroid, 5, (0, 0, 255), -1)
            cv2.putText(black_img, 'Centroid', centroid, 1, 1.1, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw the circle offset from distTransform
            cv2.circle(black_img, circle_center, int(circle_offset), (255, 0, 0), 2)
            # Draw the circle center from distTransform
            cv2.circle(black_img, circle_center, 4, (255, 255, 0), 2)
            cv2.putText(black_img, 'DF center', circle_center, 1, 1.1, (255, 255, 0), 1, cv2.LINE_AA)

            # # Draw the fit line an his center
            # cv2.line(black_img, pt1, pt2, (0, 0, 255), 2)
            # # Draw the circle center from fit line
            # cv2.circle(black_img, (int(clx), int(cly)), 5, (0, 0, 255), -1)

            cv2.imshow('Palm', black_img)

            # Draw the _contour without the point between the intersection points
            cv2.drawContours(img, [contour], 0, (255, 255, 255), 2)
            cv2.imshow('Move Center ', img)

    # Detect the intersection between the line and a hand _contour
    def findLineAndContourIntersections(self, img, contour, pta, ptb, img_limits):
        x_min, x_max, y_min, y_max = img_limits
        # Contour is a list of lists of points, cat them and squeeze out extra dimensions
        horiz_contour = self._contour_proc.contourToList(contour)
        intersections = self._geom_utils.findLineAndContourIntersections(img, horiz_contour, pta, ptb)

        return intersections

    def findPerpLineCoords(self, center, radius, line_points):
        if len(line_points) >= 1:
            inters_x = line_points[0][0]
            inters_y = line_points[0][1]
            pt_inters = (inters_x, inters_y)

            # Find a line perpendicular to a line from a point of the center circle around the hand
            # to an intersection point with the circle radius
            px1, py1, px2, py2 = self._geom_utils.findPerpLineCoords(center[0], center[1], inters_x, inters_y, radius)

            return px1, py1, px2, py2, pt_inters

        return None, None, None, None, None

    def findForearmLimits(self, img, contour, center, radius, axis_circle_intersections, draw=False):
        found = False
        inters = []

        # Find a line perpendicular to a line from a point of the center circle around the hand
        # to an intersection point with the circle radius
        px1, py1, px2, py2, pt_inters = self.findPerpLineCoords(center, radius, axis_circle_intersections)

        if px1 is not None:
            if draw:
                ints = (int(pt_inters[0]), int(pt_inters[1]))
                # Draw the point of intersection
                cv2.circle(img, ints, 5, (0, 130, 255), 2)
                # Draw the perpendicular line
                cv2.line(img, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)
                # Draw line from center to intersection
                cv2.line(img, (center[0], center[1]), ints, (0, 130, 255), 2)

            # Y threshold to filter points not useful to calculate intersections
            y_threshold = center[1] + (radius * self._percent_radius)
            inters = self.findContourAndCircleIntersections(contour, center, radius, y_threshold)

            if len(inters) > 0:
                inters = self.findRelevantIntersections(pt_inters, inters, [px1, py1], [px2, py2])
                size = len(inters)

                if size > 1:
                    found = True
                    if draw:
                        for i in range(size):
                            p = inters[i]
                            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), 2)

        return found, inters

    def distTransform(self, bg_mask, draw=False):
        max_width = bg_mask.shape[0]
        # Find out the distance transform from a _mask using Euclidian Distance Transform operation
        dist_transform = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
        # dist_transform, labels = cv2.distanceTransformWithLabels(bg_mask, cv2.DIST_L2, 5)

        # Get the maximum distance
        max_dist = dist_transform.max()
        center = None
        offset = None

        if max_dist < max_width:
            # Find out the max distance and his center found on the distance transform operation
            max_dist, x, y = self.maxDistTransf(dist_transform)
            center = (x, y)
            offset = int(max_dist * self._offset_radius)

            if draw:
                # Normalize the distance image for range = {0.0, 1.0}
                # so we can visualize and threshold it
                cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
                cv2.circle(dist_transform, center, int(max_dist), (255, 255, 255), 2)
                cv2.imshow('Distance Transform', dist_transform)

        return center, max_dist, offset

    def maxDistTransf(self, dist_transf):
        # Find out the max distance and his center found on the distance transform operation
        rows = dist_transf.shape[0]
        cols = dist_transf.shape[1]
        max_dist = 0
        x, y = 0, 0

        for i in range(rows):
            for j in range(cols):
                dist = dist_transf[i][j]
                if dist > max_dist:
                    x = j
                    y = i
                    max_dist = dist
        return int(max_dist), x, y

    def findRelevantIntersections(self, pt_inters, intersections, pt1, pt2):
        size = len(intersections)

        if size > 2:
            inters = []
            dists = []

            for i in range(size):
                pt = intersections[i]
                x = pt[0]
                y = pt[1]

                # Find the distance from the center-line-intersection point to the _contour-offset intersections
                dist, _ = self._geom_utils.calculateDistanceAndAngle(pt_inters, (x, y))
                #dist = self._geom_utils.point_line_distance(np.array(pt1), np.array(pt2), np.array(pt))
                dists.append(dist)

            if len(dists) > 1:
                zipped = zip(intersections, dists)
                z = list(zipped)

                # Using sorted and lambda
                res = sorted(z, key=lambda item: item[1])
                two_points = res[:2]

                for p, d in two_points:
                    inters.append([p[0], p[1]])

            return inters
        else:
            return intersections

    # Detect the intersection between the hand _contour and a circle with the center
    # equal to the determined center by the distance transform from the hand _mask
    # and the radius equal to the maximum distance from that center to the border.
    def findContourAndCircleIntersections(self, max_contour, circle_center, circle_radius, y_threshold):
        intersections = []
        # Limit to avoid search lines bellow the Y radius
        y_limit = circle_center[1] + circle_radius
        horiz_contour = self._contour_proc.contourToList(max_contour)
        size, _ = horiz_contour.shape

        # Run for each 2 points that represent a _contour line to detect if it has intersection with the palm hand circle
        for i in range(size):
            pt2 = tuple(horiz_contour[i])
            y2 = pt2[1]
            index = i + 1

            if index == size:
                index = 0

            pt1 = tuple(horiz_contour[index])
            y1 = pt1[1]

            # Determine if the points are in an useful range, verify the (y) threshold value and detect if the point
            # is above or below of the y_centroid + radius distance limit filtering only useful points to calculate intersections
            #if (not(y1 > y_limit and y2 > y_limit) or not(y1 < y_limit and y2 < y_limit)) and (y1 > y_threshold or y2 > y_threshold):
            if (not(y1 > y_limit and y2 > y_limit) or not(y1 < y_limit and y2 < y_limit)) :
                has_inters, point = self._geom_utils.findCircleLineIntersection(circle_center, circle_radius, pt1, pt2)

                if has_inters and point:
                    intersections = intersections + point

        return intersections

    def intersecReferences(self, inters):
        if len(inters) >= 2:
            p_right = inters[0]
            p_left = inters[1]
            (x_right, y_bottom) = p_right
            (x_left, y_top) = p_left

            if x_left > x_right:
                x_left, x_right = x_right, x_left
                p_left, p_right = p_right, p_left
            if y_top > y_bottom:
                y_top, y_bottom = y_bottom, y_top

            if p_left[1] >= p_right[1]:
                y_left_limit = y_bottom
                y_right_limit = y_top
            else:
                y_left_limit = y_top
                y_right_limit = y_bottom

            return True, x_left, x_right, y_top, y_bottom, p_left, p_right, y_left_limit, y_right_limit

        return False, None, None, None, None, None, None, None, None

    def filterForearmContour(self, contour, inters, center_x):
        filtered_contour = contour
        is_filtered_contour = False
        id_first = -1
        id_last = -1
        ret, x_left, x_right, y_top, y_bottom, p_left, p_right, y_l_lim, y_r_lim = self.intersecReferences(inters)

        if ret:
            horiz_contour = self._contour_proc.contourToList(contour)
            size, _ = horiz_contour.shape
            filtered_contour = []
            is_first = True
            index = 0

            for i in range(size):
                [x, y] = horiz_contour[i]
                is_forearm_range = x_right >= x >= x_left and y > y_top
                is_before_left = x < x_left and y > y_l_lim
                is_after_right = x > x_right and y > y_r_lim
                forearm_filter = is_forearm_range or is_before_left or is_after_right

                is_left_limit = not is_after_right and is_before_left and x < center_x
                is_right_limit = not is_before_left and is_after_right and x > center_x

                # if is_left_limit:
                #     print("[x_left > x >= center_x]= (", x_left, ">", x, "<", center_x, ")=", is_left_limit, ", x_left =", x_left, ", x_right =", x_right, ", center_x=", center_x)

                # if not forearm_filter and not is_left_limit and not is_right_limit:
                    #cv2.circle(img, (x, y), 2, (0, 255, 255), 2)

                if not forearm_filter: # and not is_left_limit and not is_right_limit:
                    is_filtered_contour = True
                    filtered_contour.append([x, y])
                    index += 1
                else:
                    if is_first:
                        is_first = False
                        id_first = index
                        id_last = index+1

        if is_filtered_contour:
            if id_first > -1:
                filtered_contour.insert(id_first, [ceil(p_left[0]), ceil(p_left[1])])

                if id_last > -1:
                    filtered_contour.insert(id_last, [ceil(p_right[0]), ceil(p_right[1])])

                #filtered_contour = self._geom_utils.sortPoints(filtered_contour)

            filtered_contour = self._contour_proc.convertPointSequenceToContour(filtered_contour)

        return filtered_contour


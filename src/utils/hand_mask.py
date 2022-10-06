import cv2
import copy
from numpy import zeros
from src.utils.pos_process import PosProcess
from src.utils.contour import Contour
from src.utils.forearm import Forearm

class HandMask:
    def __init__(self):
        self._pos_proc = PosProcess()
        self._contour_proc = Contour()
        self._forearm_proc = Forearm()

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

    def findHandMask(self, img, roi, hand, consider_invalid_hand=True, draw=False):
        if consider_invalid_hand:
            valid_hand = True
        else:
            valid_hand = False

        filtered_mask = hand.getMask()
        x_min, y_min, x_max, y_max, _ = hand.getDimensions()

        # 3-) Find the contours in the image an select the biggest one
        ####################################################################################################
        contours, max_contour, area = self._contour_proc.findContours(filtered_mask)

        # 4-) Clear the _mask removing all areas that are not in the of biggest _contour
        ####################################################################################################
        if max_contour is not None and len(max_contour) > 0:
            filtered_mask = zeros(filtered_mask.shape, dtype="uint8")
            cv2.drawContours(filtered_mask, [max_contour], 0, (255, 255, 255), -1)
            hand.setContour(max_contour)
            hand.setMask(filtered_mask)

            if draw:
                cv2.imshow('Final Mask', filtered_mask)

            # 5-) Remove the forearm  from the hand image to improve the palm hand image
            ####################################################################################################
            detect_hand = img[y_min:y_max, x_min:x_max]
            h, w = detect_hand.shape[:2]
            detect_hand = copy.deepcopy(detect_hand)
            valid_hand, circle_center, circle_radius, circle_offset, max_contour = self._forearm_proc.removeForearmFromHand(
                detect_hand, w, h, hand, consider_invalid_hand, draw)

            #print("Final valid_hand =", valid_hand)

        # 6-) Extracting background from the image by the filtered _mask
        ####################################################################################################
        if max_contour is not None and len(max_contour) > 0:
            filtered_mask = zeros(filtered_mask.shape, dtype="uint8")
            cv2.drawContours(filtered_mask, [max_contour], 0, (255, 255, 255), -1)
            hand.setContour(max_contour)
            hand.setMask(filtered_mask)

        hand_img = cv2.bitwise_and(roi, roi, mask=filtered_mask)

        if draw:
            cv2.imshow('Foreground', hand_img)

        return valid_hand, filtered_mask, max_contour, hand_img

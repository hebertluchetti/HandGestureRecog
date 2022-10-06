import cv2


class PosProcess:
    def __init__(self):
        pass


    def closingMask(self, mask, kernel, iterations):
        # Closing is reverse of Opening, Dilation followed by Erosion.
        # It is useful in closing small holes inside the foreground objects,
        #  or small black points on the object.
        # Apply a series of erosions and dilations to the _mask using an elliptical kernel
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.erode(mask, kernel, iterations=iterations)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
        return mask

    def openingMask(self, mask, kernel, iterations):
        # Opening is just another name of erosion followed by dilation.
        # It is useful in removing noise.
        # Apply a series of dilations and erosions to the _mask using an elliptical kernel
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.dilate(mask, kernel, iterations=iterations)
        mask = cv2.erode(mask, kernel, iterations=iterations)
        return mask

    # Remove the noise from the _mask applying opening and closing morphological filter and GaussianBlur filter
    def removeMaskNoise(self, fg_mask):
        # Remove the noise from the _mask applying opening and closing morphological filter
        filtered_mask = self.morphologicalFilter(fg_mask)

        # Blur the _mask to help remove noise, then apply the _mask to the frame
        filtered_mask = cv2.GaussianBlur(filtered_mask, (3, 3), 0)
        # filtered_mask = cv2.medianBlur(filtered_mask, 7)
        return filtered_mask

    def morphologicalFilter(self, mask):
        # kernel = np.ones((5, 5), np.uint8)
        iterations = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Opening is just another name of erosion followed by dilation. It is useful in removing noise.
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                   iterations)  # openingMask(_mask, kernel, iterations)

        # Closing is reverse of Opening, Dilation followed by Erosion.
        # It is useful in closing small holes inside the foreground objects, or small black points on the object.
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel,
                                   iterations)  # closingMask(opening, kernel, iterations)

        return closing

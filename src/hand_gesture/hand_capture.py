desc = '''Script to store files data images with a specific name

Usage: python hand_class_repository.py <class_name> <sample_quantity>

The code will collect <sample_quantity> number of images and store them in its directory with the specific name
A ROI the portion of the image within the box displayed will be captured and stored.

Press 's' to start/pause the image collecting process.
Press 'q' to quit.

'''

import cv2
import os
import shutil
import random
import glob
from src.utils.training_config import TrainingConfig
from src.utils.hand import Hand
from src.hand_gesture.base_segmentation import BaseSegmentation

class HandCapture(BaseSegmentation):

    def createDir(self, label_name):
        path_capture_sub_dir = self._file_utils.create_path_dir(label_name, "", self.IMAGE_DATA_SET_PATH, self.IMAGE_CAPTURE_PATH)
        self.CAPTURE_PATH = path_capture_sub_dir

    def organizeImageDirs(self, class_names, sample_quantity):
        # Organize data files into train, valid, test dirs
        img_train_path, img_test_path, img_valid_path = self.createImageDirs(self.IMAGE_DATA_SET_PATH, self.IMAGE_TRAINING_SET_PATH, self.IMAGE_TEST_SET_PATH, self.IMAGE_VALID_SET_PATH, class_names)
        img_capture_path = self._file_utils.create_dirs(self.IMAGE_DATA_SET_PATH, self.IMAGE_CAPTURE_PATH)

        print("The capture images will be copied from the folder [{}]".format(img_capture_path))
        print("The training images will be saved in the folder [{}]".format(img_train_path))
        print("The test images will be saved in the folder [{}]".format(img_test_path))
        print("The valid images will be saved in the folder [{}]".format(img_valid_path))

        self.distributeImageFiles(img_capture_path, img_train_path, img_test_path, img_valid_path, class_names, sample_quantity)

    def joinPath(self, path, files):
        return os.path.join(path, files)

    def distributeImageFiles(self, img_capture_path, img_train_path, img_test_path, img_valid_path, class_names, sample_quantity):
        # Organize data files into train, valid, test dirs
        total, training_quant, valid_quant, test_quant = self.defineFileQuantities(sample_quantity)

        if class_names is not None:
            for class_name in class_names:
                file_filter = "*"
                train_path = self._file_utils.create_path_dir(class_name, img_train_path)
                valid_path = self._file_utils.create_path_dir(class_name, img_valid_path)
                test_path = self._file_utils.create_path_dir(class_name, img_test_path)
                dir_path = self._file_utils.create_dirs(img_capture_path, class_name)
                list_files = os.listdir(dir_path)  # dir is the directory path
                number_files = len(list_files)

                if number_files > 0:
                    files = self.joinPath(dir_path, file_filter)

                    for i in random.sample(glob.glob(files), training_quant):
                        shutil.move(i, train_path)
                    for i in random.sample(glob.glob(files), valid_quant):
                        shutil.move(i, valid_path)
                    for i in random.sample(glob.glob(files), test_quant):
                        shutil.move(i, test_path)
                else:
                    self.logger.info("There not image file i the directory:", dir_path)

    def createImageDirs(self, img_data_path, img_train_path, img_test_path, img_valid_path, class_names):
        path_train_dir = self._file_utils.create_dirs(img_data_path, img_train_path)
        path_test_dir = self._file_utils.create_dirs(img_data_path, img_test_path)
        path_valid_dir = self._file_utils.create_dirs(img_data_path, img_valid_path)

        if class_names is not None:
            for class_name in class_names:
                self._file_utils.create_sub_dir(path_train_dir, class_name)
                self._file_utils.create_sub_dir(path_test_dir, class_name)
                self._file_utils.create_sub_dir(path_valid_dir, class_name)

        return path_train_dir, path_test_dir, path_valid_dir

    def saveImage(self, hand_img, path, count):
        save_path = os.path.join(path, '{}{}'.format(count + 1, self.JPG_EXT))
        cv2.imwrite(save_path, hand_img)

    def resetBackgroundModel(self, label_name):
        self._is_bg_captured = False
        self._bg_sub.resetBackgroundModel()
        # Create the repository directory to store the images to be trained
        self.createDir(label_name)

    def saveClassImageFile(self, frame, hand_img, path, class_name, pt_msg, count, quantity, can_save):
        # ------------------------------------------------------------------------------------------------------
        #  Capture class image for training and test - Begin
        # ------------------------------------------------------------------------------------------------------
        stop_capture = False
        msg_aux = "[" + class_name + "]"

        # Capture the training images with
        if count < quantity:
            msg = "Capture class"+msg_aux
            c = count
            count += 1
        else:
            stop_capture = True

        if can_save:
            # Save the image with the hand roi area in the repository
            self.saveImage(hand_img, path, c)
        else:
            msg = "Discarded frame " + msg_aux

        font = cv2.FONT_HERSHEY_SIMPLEX
        c = c + 1
        cv2.putText(frame, msg + ": {}".format(c), pt_msg, font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        return stop_capture, count

    def defineFileQuantities(self, sample_quantity):
        training_quant = sample_quantity
        test_quant = int(sample_quantity * 0.20)
        valid_quant = int(sample_quantity * 0.25)
        total = training_quant + test_quant + valid_quant
        print("Total=", total, ", train num=", training_quant, ", test num=", test_quant, ", valid num=", valid_quant)
        return total, training_quant, valid_quant, test_quant

    def captureImages(self, class_name, sample_quantity):
        print("This code collect", sample_quantity, "images and store them in its directory with the specific name:[", class_name, "]")
        print("A portion of the image within the box displayed (ROI) will be captured and stored.")
        print("Press 'b' to capture the background.\nPress 'r' to reset the background")
        print("Press 's' to start/pause the image collecting process.\nPress 'q' to quit.")
        print("-------------------------------------------------------------------------------------------------------\n")
        # Camera capture
        self._cap = self._cap_conf.getCapture()
        self._is_bg_captured = False  # bool, whether the background captured

        win_name = "Capturing [" + class_name + "] images"
        msg = self.BACKGROUND_MSG
        font = cv2.FONT_HERSHEY_SIMPLEX
        hand_class_dir = self._file_utils.create_dirs(TrainingConfig.HAND_CLASS_PATH)
        start, valid_hand, count, test_count = False, False, 0, 0

        # Define the quantity of image files is needed to separate train, valid and test sets
        quantity, training_quant, valid_quant, test_quant = self.defineFileQuantities(sample_quantity)

        # Defines the area of interest
        (x_min, x_max, y_min, y_max, area, roi_color), width, height = self._cap_conf.defineHandROI()
        # Define the message position on the screen
        x_ref, y_ref = int(width * 0.005), int(height * 0.025)
        # Define the class message position on the screen
        x_class, y_class = x_ref, y_ref * 2
        # Define the counter message position on the screen
        x_count, y_count = x_ref, y_ref * 4

        # Create a named window and move it to (x_cap_win, y_cap_win)
        self.createCaptureWin(win_name, width, height)
        # Create the repository directory to store the images to be trained
        self.createDir(class_name)

        # Create the hand object toi keep the hand information
        hand = Hand()
        hand.setDimensions(x_min, y_min, x_max, y_max)
        capture_all_frames = False

        try:
            self.showAllClasses(hand_class_dir, font)
            while self._cap.isOpened():
                ret, frame = self._cap.read()

                if not ret:
                    continue
                elif count == quantity:
                    capture_all_frames = True
                    break

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # ROI rectangle
                # Define area to capture the hand movement
                self.drawHandROI(frame, x_min, x_max, y_min, y_max, roi_color)
                cv2.putText(frame, msg, (x_ref, y_ref), 1, 1.4, self.text_color, 1, cv2.LINE_AA)
                cv2.putText(frame, self.QUIT_MSG, (x_class, y_class), 1, 1.4, self.text_color, 1, cv2.LINE_AA)
                cv2.putText(frame, win_name, (x_class, y_ref*3), 1, 1.4, self.text_color, 1, cv2.LINE_AA)

                # ------------------------------------------------------------------------------------------------------
                #  Background subtraction operation - Begin
                #------------------------------------------------------------------------------------------------------
                if self._is_bg_captured:  # this part wont run until background captured
                    result, valid_hand, hand_img, _ = self.getHandImage(hand, frame, x_min, y_min, x_max, y_max,
                                                                        False)
                    if not result:
                        continue
                    cv2.imshow('Foreground', hand_img)

                # ------------------------------------------------------------------------------------------------------
                #  Capture class image for training and test - Begin
                # ------------------------------------------------------------------------------------------------------
                if start:
                    if valid_hand:
                        # In case the background subtraction is not initialized
                        # the original frame is used to be captured
                        if not self._is_bg_captured:
                            hand_img = frame[y_min:y_max, x_min:x_max]  # clip the ROI

                        # 3-) Save the image with the hand roi _area in the repository
                        stop_capture, count = self.saveClassImageFile(frame, hand_img, self.CAPTURE_PATH, class_name, (x_count, y_count), count, quantity, True)
                        if stop_capture:
                            break
                    else:
                        self.saveClassImageFile(frame, None, self.CAPTURE_PATH, class_name, (x_count, y_count), count, quantity, False)

                cv2.imshow(win_name, frame)

                key = cv2.waitKey(10)
                if key == ord('s'):  # press 's' to capture the training and test images
                    start = not start
                elif key == ord('b'):  # press 'b' to capture the background
                    if not self._is_bg_captured:
                        self.createBackgroundModel(self._bg_sub_type, self._history, self._bg_sub_threshold, self._detect_shadows, self._learning_rate)
                        msg = self.START_MSG
                        print('Background Model created !')
                    else:
                        print('Background Model already created !')
                elif key == ord('r'):  # press 'r' to reset the background
                    self.resetBackgroundModel(class_name)
                    msg = self.BACKGROUND_MSG
                    print('Reset Background !')
                elif key == ord('q') or key == 27:  # press ESC or 'q'to exitxx
                    break

            print("\n=> {} training image(s) saved to {}".format(count, self.TRAINING_PATH))
            print("\n=> {} test image(s) saved to {}".format(test_count, self.TEST_PATH))
        finally:
            self.releaseAll(win_name)

        return capture_all_frames

def main():
    sample_quantity = 250
    organize_files = False
    class_names = HandCapture.CLASS_MAP

    print("-------------------------------------------------------------------------------------------------------")
    print("\nStarting the process ...")
    print("\nClasses that will be created =>", class_names)

    if class_names is not None and sample_quantity is not None:
        capture_all_frames = True
        for class_name in class_names:
            cap = HandCapture()
            print("\nCapture for the class name [", class_name, "], capturing [", sample_quantity, "] images")

            sample_quantity = int(sample_quantity)
            cap_all = cap.captureImages(class_name, sample_quantity)

            if not cap_all:
                capture_all_frames = False
                break

        if organize_files and capture_all_frames:
            cap.organizeImageDirs(class_names, sample_quantity)
    else:
        print("\nInvalid parameter! Try again.\nGesture class name: [{}], Sample quantity: [{}]".format(class_names,
                                                                                                        sample_quantity))


if __name__ == '__main__':
    main()




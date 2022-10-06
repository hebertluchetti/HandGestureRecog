desc = '''
            Script to predict hand images with a neural networked trained
            Press 's' to start/pause the image collecting process.
            Press 'b' to start the background segmentation process wait 5 secs to show the hand"
            Press 'r' to rest the background segmentation process"
            Press 'ESC' to quit.
        '''
import cv2
from numpy import argmax, array
from src.utils.training_config import TrainingConfig
from src.utils.hand import Hand
from src.hand_gesture.base_segmentation import BaseSegmentation
from src.hand_gesture.bar_h_graph import BarHGraph, PredictData

class HandRecognition(BaseSegmentation):
    def __init__(self, source):
        super().__init__(source)
        self._bar_h_graph_active = False
        self._pred_data = PredictData(self.CLASS_NAMES)
        self._bar_h_graph = BarHGraph(self._pred_data)

    def pause(self):
        self._bar_h_graph_active = False

    def predict(self, model, hand_img):
        img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
        img = img.reshape(1, self.WIDTH, self.HEIGHT, 3)
        return self.predictRgbImageVgg(model, img)
        #return = self.predictRgbImage(model, img)

    def predictRgbImage(self, model, image):
        predictions = model.predict_classes(image)
        class_name, score = self.findPredicted(predictions)
        return class_name, score, predictions

    def predictRgbImageVgg(self, model, image):
        image = array(image, dtype='float32')
        image /= 255
        predictions = model.predict(image)
        class_name, score = self.findPredicted(predictions)
        return class_name, score, predictions

    def findPredicted(self, predictions):
        class_name, score = None, None

        if predictions is not None and len(predictions) > 0:
            # one_hot_encoded
            rounded_predictions = argmax(predictions, axis=-1)
            class_index = rounded_predictions[0]
            max_val = predictions[0][class_index]

            class_name = self.getGestureName(class_index)
            score = float("%0.2f" % (max_val * 100))
            self.logger.info("Predicted: {}, Score:{}".format(class_name, score))

            if score < 50:
                class_name = self.UNKNOWN_CLASS

        return class_name, score

    def choosePredictedClass(self, class_name, score, hand_class_dir, msg_point, roi_w, roi_h, font):
        pred_msg = f"Prediction:[{class_name}]"

        if class_name != self.UNKNOWN_CLASS:
            pred_msg = f"{pred_msg} ({score}%)"
            pred_color = self.text_color
        else:
            pred_color = self.red_color

        hand_class_img = self._dict_classes[class_name]
        hand_class_img = cv2.resize(hand_class_img, (roi_w, roi_h))
        cv2.putText(hand_class_img, pred_msg, msg_point, font, 0.5, pred_color, 1, cv2.LINE_AA)

        return hand_class_img

    def controlPredictionInfo(self, start, valid_hand, predictions, class_name, hand_class_img, frame):
        # Control the bar graph with the predictions values
        # -----------------------------------------------------
        if start:
            if valid_hand:
                if not self._bar_h_graph_active:
                    self._bar_h_graph.show()
                    self._bar_h_graph_active = True

                if self._bar_h_graph_active and predictions is not None and len(predictions) > 0:
                    data = predictions[0] * 100
                    self._pred_data.update(data)
                    #self.logger.info("Predictions: %s ", data)
            elif self._bar_h_graph_active:
                self._pred_data.update(self._pred_data.default)
        else:
            self._pred_data.update(self._pred_data.default)
        # -----------------------------------------------------

    def recognize(self):
        self.logger.info("A portion of the image within the box displayed (ROI) will be captured and searched in a neural network.")
        self.logger.info("Press 'b' to capture the background.")
        self.logger.info("Press 'r' to reset the background")
        self.logger.info("Press 's' to start/pause the image collecting process.")
        self.logger.info("Press 'q' to quit.")
        self.logger.info("-------------------------------------------------------------------------------------------------------\n")
        # Camera capture
        self._cap = self._cap_conf.getCapture()
        self._is_bg_captured = False  # bool, whether the background captured

        win_name = "Recognizing hand gestures"
        msg = self.BACKGROUND_MSG
        font = cv2.FONT_HERSHEY_SIMPLEX
        hand_class_dir = self._file_utils.create_dirs(TrainingConfig.HAND_CLASS_PATH)
        start, valid_hand = False, False

        # Defines the area of interest
        (x_min, x_max, y_min, y_max, area, roi_color), width, height = self._cap_conf.defineHandROI()
        roi_w, roi_h = (x_max - x_min), (y_max - y_min)
        # Define the message position on the screen
        x_ref, y_ref = int(width * 0.005), int(height * 0.025)
        # Define the class message position on the screen
        x_class, y_class = x_ref, y_ref * 2
        # Define the counter message position on the screen
        x_count, y_count = x_ref, y_ref * 22
        msg_point = (10, int(roi_h - (y_ref / 2)))

        # Create a named window and move it to (x_cap_win, y_cap_win)
        self.createCaptureWin(win_name, width, height)

        # Create the hand object toi keep the hand information
        hand = Hand()
        hand.setDimensions(x_min, y_min, x_max, y_max)

        try:
            self.showAllClasses(hand_class_dir, font)
            model, _ = self.loadModels()

            while self._cap.isOpened():
                ret, frame = self._cap.read()

                if not ret:
                    continue

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # ROI rectangle
                # Define area to capture the hand movement
                self.drawHandROI(frame, x_min, x_max, y_min, y_max, roi_color)
                cv2.putText(frame, msg, (x_ref, y_ref), 1, 1.4, self.text_color, 1, cv2.LINE_AA)
                cv2.putText(frame, self.QUIT_MSG, (x_class, y_class), 1, 1.4, self.text_color, 1, cv2.LINE_AA)
                cv2.putText(frame, win_name, (x_class, y_ref*3), 1, 1.4, self.text_color, 1, cv2.LINE_AA)

                # ------------------------------------------------------------------------------------------------------
                #  Background subtraction operation
                #------------------------------------------------------------------------------------------------------
                hand_img = None
                if self._is_bg_captured:  # this part won't run until background captured
                    result, valid_hand, hand_img, _ = self.getHandImage(hand, frame, x_min, y_min, x_max, y_max, False)
                    if not result:
                        continue

                # ------------------------------------------------------------------------------------------------------
                #  Capture class image for prediction
                # ------------------------------------------------------------------------------------------------------
                class_name = self.UNKNOWN_CLASS
                score, predictions = None, None

                if self._is_bg_captured:
                    if start:
                        if valid_hand:
                            h_img = frame[y_min:y_max, x_min:x_max]  # clip the ROI
                            # Predict the move made
                            class_name, score, predictions = self.predict(model, h_img)
                            cv2.putText(frame, win_name, (x_class, y_ref * 3), 1, 1.4, self.text_color, 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Prediction: {class_name} ({score}%)", (x_count, y_count), font, 1, self.text_color, 1, cv2.LINE_AA)

                    # Select the predicted hand class
                    hand_class_img = self.choosePredictedClass(class_name, score, hand_class_dir, msg_point, roi_w, roi_h, font)

                    if hand_img is None:
                        im_h = hand_class_img
                    else:
                        im_h = cv2.hconcat((hand_img, hand_class_img))

                    # Control the bar graph with the predictions values
                    #-----------------------------------------------------
                    self.controlPredictionInfo(start, valid_hand, predictions, class_name, hand_class_img, frame)
                    # -----------------------------------------------------

                    cv2.imshow("Prediction", im_h)

                cv2.imshow(win_name, frame)

                key = cv2.waitKey(10)
                if key == ord('s'):  # press 's' to capture the training and test images
                    start = not start
                elif key == ord('b'):  # press 'b' to capture the background
                    if not self._is_bg_captured:
                        self.createBackgroundModel(self._bg_sub_type, self._history, self._bg_sub_threshold, self._detect_shadows, self._learning_rate)
                        msg = self.START_MSG
                        self.logger.info('Background Model created !')
                    else:
                        self.logger.info('Background Model already created !')
                elif key == ord('r'):  # press 'r' to reset the background
                    self.resetBackgroundModel()
                    start = not start
                    msg = self.BACKGROUND_MSG
                    self.logger.info('Reset Background !')
                elif key == ord('q') or key == 27:  # press ESC or 'q'to exitxx
                    break

        finally:
            self.releaseAll(win_name)

def main():
    reg = HandRecognition(0)
    class_names = reg.CLASS_MAP
    reg.logger.info("-------------------------------------------------------------------------------------------------------")
    reg.logger.info("Starting the process ...")
    reg.logger.info("The classes that can be recognized are:", class_names)
    reg.recognize()


if __name__ == '__main__':
    main()




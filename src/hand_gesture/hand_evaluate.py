import numpy as np
from os import path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils.training_config import TrainingConfig
from src.utils.file_utils import FileUtils
from src.utils.graph_plot import GraphPlot


class HandEvaluate:
    def __init__(self):
        self._init()

    def _init(self):
        self._file_utils = FileUtils()
        self._train_conf = TrainingConfig()
        self._graph = GraphPlot()

        self.IMAGE_DATA_SET_PATH = TrainingConfig.IMAGE_DATA_SET_PATH
        self.IMAGE_TRAINING_SET_PATH = TrainingConfig.IMAGE_TRAINING_SET_PATH
        self.IMAGE_TEST_SET_PATH = TrainingConfig.IMAGE_TEST_SET_PATH
        self.IMAGE_VALID_SET_PATH = TrainingConfig.IMAGE_VALID_SET_PATH
        self.IMAGE_CLASS_PATH = None
        self.IMAGE_TEST_CLASS_PATH = None

        self.IMG_SAVE_PATH = TrainingConfig.IMAGE_DATA_SET_PATH
        self.FILE_NAME_MODEL = TrainingConfig.FILE_NAME_MODEL
        self.FILE_WEIGHT_MODEL = TrainingConfig.FILE_WEIGHT_MODEL
        self.CLASS_MAP = TrainingConfig.CLASS_MAP_ITEMS
        self.NUM_CLASSES = len(self.CLASS_MAP)
        self.CLASS_NAMES = list(self.CLASS_MAP.keys())
        self.WIDTH, self.HEIGHT = TrainingConfig.VGG16_WIDTH, TrainingConfig.VGG16_HEIGHT

        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 5

    def getGestureName(self, val):
      class_names = self.CLASS_NAMES
      return class_names[val]

    def loadModels(self):
        source_dir = TrainingConfig.MODEL_PATH
        model_path = self._file_utils.create_dirs(source_dir)
        load_path = path.join(model_path, self.FILE_NAME_MODEL)
        load_weight_path = path.join(model_path, self.FILE_WEIGHT_MODEL)

        print("The model file will be loaded from the folder [{}]".format(load_path))
        print("The weights file will be loaded from the folder [{}]".format(load_weight_path))

        if path.exists(load_path):
            print("The model file exists in the directory:" + load_path)

            # Load the model
            model = load_model(load_path)
            weights = None

            if path.exists(load_weight_path):
              print("The weight model file exists in the directory:" + load_weight_path)
              weights = model.load_weights(load_weight_path)
            else:
              print("The weight model file not exists in the directory:" + load_weight_path)

            return model, weights
        else:
            print("The model not exists in the directory:" + load_path)
        return None, None

    def predict(self):
        class_names = self.CLASS_NAMES
        model, _ = self.loadModels()

        if model is not None:
            test_path = self._file_utils.get_path(self.IMAGE_DATA_SET_PATH, self.IMAGE_TEST_SET_PATH)

            print(test_path)

            # This is the preprocessing that normalizing the image values to improve the processinge
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            print(class_names)
            test_batches = test_datagen.flow_from_directory(directory=test_path, target_size=(self.WIDTH, self.HEIGHT),
                                                            classes=class_names, batch_size=self.batch_size,
                                                            shuffle=False)

            setps = test_batches.samples / self.batch_size
            predictions = model.predict_generator(test_batches, setps)
            self.evaluateModel(model, test_batches, self.batch_size)
            self._graph.showConfusionMatrix(test_batches.classes, predictions, class_names, 'Confusion Matrix')


    def evaluateModel(self, model, generator, batch_size):
        score = model.evaluate_generator(generator=generator,  # Generator yielding tuples
                                       steps=generator.samples // batch_size,
                                       # number of steps (batches of samples) to yield from generator before stopping
                                       max_queue_size=10,  # maximum size for the generator queue
                                       workers=1,
                                       # maximum number of processes to spin up when using process based threading
                                       use_multiprocessing=False,  # whether to use process-based threading
                                       verbose=0)
        loss, accuracy = score[0], score[1]
        print("\nTest accuracy: loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    def show(self, predictions):
        for i in range(len(predictions)):
          result = predictions[i]
          answer = np.argmax(result)
          move_name = self.getGestureName(answer)
          print("Predicted: {}".format(move_name))


def main():
    pred = HandEvaluate()
    pred.predict()


if __name__ == '__main__':
    main()

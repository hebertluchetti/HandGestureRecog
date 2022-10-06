import cv2
import os
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import applications

from src.utils.training_config import TrainingConfig
from src.utils.file_utils import FileUtils
from src.utils.graph_plot import GraphPlot

class BaseTraining:
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
        self.MODEL_PATH = TrainingConfig.MODEL_PATH

        self.IMG_SAVE_PATH = TrainingConfig.IMAGE_DATA_SET_PATH
        self.FILE_NAME_MODEL = TrainingConfig.FILE_NAME_MODEL
        self.FILE_WEIGHT_MODEL = TrainingConfig.FILE_WEIGHT_MODEL
        self.CLASS_MAP = TrainingConfig.CLASS_MAP_ITEMS
        self.NUM_CLASSES = len(self.CLASS_MAP)
        self.CHANNELS = TrainingConfig.IMAGE_CHANNELS
        self.WIDTH, self.HEIGHT = TrainingConfig.VGG16_WIDTH, TrainingConfig.VGG16_HEIGHT

        # if batch_size = 10 and the n umber of test set files i the test_set directory is 50
        # the number of sets for predicting must be only 5 (step=5)
        # batch_size = 10 the step must be 5 for 50 test files
        # so  setps = len(test_batches) / batch_size
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 5

        self.filterConv1 = 32
        self.filterConv2 = 64
        self.filterConv3 = 128
        self.units = 512
        self.filter_size1 = (3, 3)
        self.filter_size2 = (2, 2)
        self.pool_size = (2, 2)

    def mapper(self, val):
        return self.CLASS_MAP[val]

    def getPaths(self):
        train_path = self._file_utils.get_path(self.IMAGE_DATA_SET_PATH, self.IMAGE_TRAINING_SET_PATH)
        valid_path = self._file_utils.get_path(self.IMAGE_DATA_SET_PATH, self.IMAGE_VALID_SET_PATH)
        test_path = self._file_utils.get_path(self.IMAGE_DATA_SET_PATH, self.IMAGE_TEST_SET_PATH)
        return train_path, valid_path, test_path

    def loadDir(self, dataset):
        path = self._file_utils.get_path(self.IMAGE_DATA_SET_PATH, self.IMAGE_TRAINING_SET_PATH)
        return self.loadImageDataset(path, dataset)

    def loadImageDataset(self, path, dataset):
        width, height = self.WIDTH, self.HEIGHT

        for directory in os.listdir(path):
            dir_path = os.path.join(path, directory)

            if not os.path.isdir(dir_path):
                continue

            for item in os.listdir(dir_path):
                # to make sure no hidden files get in our way
                if item.startswith("."):
                    continue
                img = cv2.imread(os.path.join(dir_path, item))

                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (width, height))
                    dataset.append([img, directory])

    def saveModels(self, model):
        source_dir = self.MODEL_PATH
        model_path = self._file_utils.create_dirs(source_dir)
        save_path = os.path.join(model_path, self.FILE_NAME_MODEL)
        save_weight_path = os.path.join(model_path, self.FILE_WEIGHT_MODEL)
        dt_str = self._file_utils.cur_datetime()

        if os.path.exists(save_path):
            file_name = '{}_{}'.format(dt_str, self.FILE_NAME_MODEL)
            new_path = os.path.join(model_path, file_name)
            self._file_utils.rename(save_path, new_path)
            print("The old model file was renamed to [{}]".format(new_path))

        if os.path.exists(save_weight_path):
            file_name = '{}_{}'.format(dt_str, self.FILE_WEIGHT_MODEL)
            new_path = os.path.join(model_path, file_name)
            self._file_utils.rename(save_weight_path, new_path)
            print("The old weights file was renamed to [{}]".format(new_path))

        # save the model
        model.save(save_path)
        model.save_weights(save_weight_path)
        print("The model file was created: [{}]".format(save_path))
        print("The weights file was created: [{}]".format(save_weight_path))

    def showAccuracy(self, history, epochs=None):
        return self._graph.showAccuracy(history, epochs)

    def plotImages(self, images_arr):
        return self._graph.plotImages(images_arr)

    def getOwnModel(self, class_size):
        # Parameters
        width, height = self.WIDTH, self.HEIGHT
        channels = self.CHANNELS

        model = Sequential()
        model.add(Convolution2D(self.filterConv1, self.filter_size1, padding ="same", input_shape=(width, height, channels), activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Convolution2D(self.filterConv2, self.filter_size2, padding ="same"))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model = self.createClassifierLayers(model, class_size)

        return model

    def createClassifierLayers(self, model, class_size):
        model.add(Flatten())

        # add a 2 dense layers with relu ativation  layer with 256 and 128 hidden units
        model.add(Dense(256, activation='relu', name='fc1'))
        #model.add(Dense(128, activation='relu', name='fc2'))

        # add a Dropout layer
        model.add(Dropout(0.5))

        # add a 1 dense layers with relu ativation  layer with 64 hidden units
        #model.add(Dense(64, activation='relu', name='fc3'))

        # add our new softmax layer with class_size hidden units
        model.add(Dense(class_size, activation='softmax', name='output'))

        return model

    def getModel(self, class_size):
        width, height = self.WIDTH, self.HEIGHT
        channels = self.CHANNELS

        model = Sequential()
        model.add(Convolution2D(self.filterConv1, self.filter_size1, activation='relu', input_shape=(width, height, channels)))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(self.pool_size))
        #model.add(Dropout(0.25))

        model.add(Convolution2D(self.filterConv2, self.filter_size1, activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(self.pool_size))
        #model.add(Dropout(0.25))

        model.add(Convolution2D(self.filterConv3, self.filter_size1, activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(self.pool_size))
        #model.add(Dropout(0.25))

        model.add(Convolution2D(self.filterConv3, self.filter_size1, activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(self.pool_size))
        #model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(self.units, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(class_size, activation='softmax'))# Number of classes

        model = self.createClassifierLayers(model, class_size)
        model.summary()

        return model

    def compileModelLayers(self, class_names, number_of_classes):
        # Get the image paths
        train_path, valid_path, test_path = self.getPaths()

        # This is the preprocessing that was used on the original training data, and therefore,
        # this is the way we need to process images before passing them to VGG16 or a fine-tuned VGG16 mode
        train_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input, rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
        valid_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input, rescale=1. / 255)

        # train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
        # valid_datagen = ImageDataGenerator(rescale=1. / 255)

        # https://deeplizard.com/learn/video/LhEMXbjGV_4
        # Keras' ImageDataGenerator class create batches of data from the train, valid, and test directories.
        # ImageDataGenerator.flow_from_directory() creates a DirectoryIterator, which generates batches of normalized
        # tensor image data from the respective data directories.
        train_batches = train_datagen.flow_from_directory(directory=train_path, target_size=(self.WIDTH, self.HEIGHT),
                                                          classes=class_names, batch_size=self.batch_size,
                                                          class_mode='categorical')
        valid_batches = valid_datagen.flow_from_directory(directory=valid_path, target_size=(self.WIDTH, self.HEIGHT),
                                                          classes=class_names, batch_size=self.batch_size,
                                                          class_mode='categorical')

        # Create the model network
        cnn_model = self.getModel(number_of_classes)

        # Compile the models using the Adam optimizer with a learning rate of 0.0001,
        # categorical_crossentropy as our loss, and ‘accuracy’ as our metric.
        cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])

        return cnn_model, train_batches, valid_batches

    def trainModel(self, cnn_model, train_batches, valid_batches):
        # The size of these batchs are determined by the batch_size that set when teh train_batches were created.
        steps_per_epoch = train_batches.samples / self.batch_size
        validation_steps = valid_batches.samples / self.batch_size

        # fit the model
        history = cnn_model.fit_generator(
            train_batches,
            validation_data=valid_batches,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )

        return cnn_model, history

    def training(self):
        K.clear_session()

        # Create the model network
        size = self.NUM_CLASSES
        class_names = list(self.CLASS_MAP.keys())

        # Create batches of data from the train, valid, and test directories.
        # Generate batches of normalized tensor image data from the respective data directories.
        # Compile the models using the Adam optimizer with a learning rate of 0.0001,
        # categorical_crossentropy as our loss, and ‘accuracy’ as our metric.
        cnn_model, train_batches, valid_batches = self.compileModelLayers(class_names, size)

        t = time.time()
        # fit the model
        cnn_model, history = self.trainModel(cnn_model, train_batches, valid_batches)
        print('Training time: %s' % (time.time() - t))

        self.saveModels(cnn_model)
        self.showAccuracy(history, self.epochs)

import numpy as np
from sklearn.utils import shuffle

from src.lib.keras_squeezenet.squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf

from src.hand_gesture.base_training import BaseTraining

class HandTrainingSqueezeNet(BaseTraining):
    def __init__(self):
        super().__init__()
        self.bnmomemtum = 0.9
        self.learning_rate = 0.0001
        self.epochs = 5
        self.WIDTH, self.HEIGHT = 227, 227
        self.IMAGE_SIZE = self.WIDTH, self.HEIGHT

    def fire(self, x, squeeze, expand):
        self.bnmomemtum
        y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
        y = tf.keras.layers.BatchNormalization(momentum=self.bnmomemtum)(y)
        y1 = tf.keras.layers.Conv2D(filters=expand // 2, kernel_size=1, activation='relu', padding='same')(y)
        y1 = tf.keras.layers.BatchNormalization(momentum=self.bnmomemtum)(y1)
        y3 = tf.keras.layers.Conv2D(filters=expand // 2, kernel_size=3, activation='relu', padding='same')(y)
        y3 = tf.keras.layers.BatchNormalization(momentum=self.bnmomemtum)(y3)
        return tf.keras.layers.concatenate([y1, y3])

    def fire_module(self, squeeze, expand):
        return lambda x: self.fire(x, squeeze, expand)

    def squeezeNetModel(self):
        x = tf.keras.layers.Input(shape=[*self.IMAGE_SIZE, 3])  # input is 192x192 pixels RGB

        y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)
        y = tf.keras.layers.BatchNormalization(momentum=self.bnmomemtum)(y)
        y = self.fire_module(24, 48)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = self.fire_module(48, 96)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = self.fire_module(64, 128)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = self.fire_module(48, 96)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = self.fire_module(24, 48)(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dense(5, activation='softmax')(y)

        model = tf.keras.Model(x, y)
        return model

    def getModel(self, size):
        width, height = self.WIDTH, self.HEIGHT
        channels = self.IMAGE_CHANNELS
        input_shape = width, height, channels
        # batch_size = 256
        # lr_rate = 0.0001
        # out_classes = 200
        # is_train = True
        # max_iter = 25000
        #sq_net = SqueezeNet(input_shape, out_classes, lr_rate, is_train)

        model = Sequential([
            SqueezeNet(input_shape=input_shape, include_top=False),
            Dropout(0.5),
            Convolution2D(self.NUM_CLASSES, (1, 1), padding='valid'),
            Activation('relu'),
            GlobalAveragePooling2D(),
            Activation('softmax')
        ])
        return model

    def training(self):
        img_save_path, file_name_model = self.IMG_SAVE_PATH, self.FILE_NAME_MODEL

        # load images from the directory
        dataset = []
        self.loadDir(dataset)
        '''
        dataset = [
            [[...], 'rock'],
            [[...], 'paper'],
            ...
        ]
        '''
        if len(dataset) == 0:
            print("No image found in the path ["+img_save_path+"]")
            return

        data, labels = zip(*dataset)
        labels = list(map(self.mapper, labels))

        '''
        labels: rock,paper,paper,scissors,rock...
        one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
        '''
        # one hot encode the labels
        labels = np_utils.to_categorical(labels)
        # valid_imgs, labels = next(valid_batches)

        # define the model
        model = self.getModel(None) # self.squeezeNetModel()
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        train_labels = np.array(labels)
        training_dataset = np.array(data)
        train_labels, training_dataset = shuffle(train_labels, training_dataset)

        # start training
        model.fit(training_dataset, train_labels, epochs=self.epochs)

        # save the model for later use
        model.save(file_name_model)

def main():
    train = HandTrainingSqueezeNet()
    train.training()

if __name__ == '__main__':
    main()
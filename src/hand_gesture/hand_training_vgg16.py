
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications

from src.hand_gesture.base_training import BaseTraining

class HandTrainingVgg16(BaseTraining):
    def __init__(self):
        super().__init__()
        self.epochs = 5

    def getVGG16ModelWithNewFCClassfier(self, class_size):
        width, height = self.WIDTH, self.HEIGHT
        channels = self.CHANNELS
        # self.FILE_NAME_MODEL = TrainingConfig.FILE_NAME_MODEL_ROCK_PAPER_SCISSORS
        # self.FILE_WEIGHT_MODEL = TrainingConfig.FILE_NAME_WEIGHT_ROCK_PAPER_SCISSORS

        # https://livebook.manning.com/book/grokking-deep-learning-for-computer-vision/chapter-6/v-8/36
        # download the model’s pretrained weights and save it in the variable base_model
        # note that here we specified to keras to download the imagenet weights
        # include_top is false to ignore the FC classifier part on top of the model
        base_model = applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(width, height, channels))

        base_model.summary()

        # Iterates over each of the layers in our new Sequential model and set them to be non-trainable.
        # This freezes the weights and other trainable parameters in each layer so that they will not be trained or
        # updated when we later pass in other images from different classes.
        for layer in base_model.layers:
            layer.trainable = False

        # Add top layer
        # save the output of the last layer to be the input of the next layer
        last_layer = base_model.output
        # flatten the classifier input which is output of the last layer of VGG16 model
        x = Flatten(name='flatten')(last_layer)

        # add a 2 dense layers with relu ativation  layer with 128 hidden units
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)

        # # add a Dropout layer
        # x = Dropout(0.5)(x)
        # # add a 1 dense layers with relu ativation  layer with 64 hidden units
        # x = Dense(64, activation='relu', name='fc3')(x)

        # add our new softmax layer with class_size hidden units
        predictions = Dense(class_size, activation='softmax', name='output')(x)
        # instantiate a new_model using keras’s Model class
        new_model = Model(inputs=base_model.input, outputs=predictions)

        # print the new_model summary
        new_model.summary()

        return new_model

    def getVGG16Model(self, class_size):
        #https://github.com/theclassofai/Deep_Transfer_Learning/blob/master/Exercise1_Classification_DogvsCat_CNN_.ipynb
        # Import the VGG16 model from Keras. An internet connection is needed to download this model.
        #vgg = applications.vgg16.VGG16()
        width, height = self.WIDTH, self.HEIGHT
        channels = self.CHANNELS
        vgg = applications.vgg16.VGG16(weights="imagenet", input_shape=(width, height, channels))
        #vgg.summary()

        model = Sequential()

        # Replicates the entire vgg16_model (excluding the output layer) to a new Sequential model,
        # which we've just given the name model.
        for layer in vgg.layers[:-1]: # The last layer
            model.add(layer)

        # Iterates over each of the layers in our new Sequential model and set them to be non-trainable.
        # This freezes the weights and other trainable parameters in each layer so that they will not be trained or
        # updated when we later pass in other images from different classes.
        for layer in model.layers:
            layer.trainable = False

        #model.add(Dropout(0.5))
        model.add(Dense(class_size, activation='softmax', name='output'))
        #model.summary()

        return model

    def getModel(self, class_size):
        return self.getVGG16ModelWithNewFCClassfier(class_size)
        #return self.getVGG16Model(class_size)

    #https://www.youtube.com/watch?v=09_gmAeHIW0&feature=youtu.be
    # def training(self):
    #     # Clear any tensorflow open session before start another one
    #     K.clear_session()
    #
    #     # Create the model network
    #     size = self.NUM_CLASSES
    #     class_names = list(self.CLASS_MAP.keys())
    #
    #     # Create batches of data from the train, valid, and test directories.
    #     # Generate batches of normalized tensor image data from the respective data directories.
    #     # Compile the models using the Adam optimizer with a learning rate of 0.0001,
    #     # categorical_crossentropy as our loss, and ‘accuracy’ as our metric.
    #     cnn_model, train_batches, valid_batches = self.compileModelLayers(class_names, size)
    #
    #     t = time.time()
    #     # fit the model
    #     cnn_model, history = self.trainModel(cnn_model, train_batches, valid_batches)
    #     print('Training time: %s' % (time.time() - t))
    #
    #     self.saveModels(cnn_model)
    #     self.showAccuracy(history, self.epochs)

def main():
    train = HandTrainingVgg16()
    train.training()

if __name__ == '__main__':
    main()
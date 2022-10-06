
class TrainingConfig:
    IMAGE_DATA_SET_PATH = 'image_data'
    IMAGE_TRAINING_SET_PATH = 'training_set'
    IMAGE_TEST_SET_PATH = 'test_set'
    IMAGE_VALID_SET_PATH = 'valid_set'
    IMAGE_CAPTURE_PATH = 'capture'
    MODEL_PATH = 'model'
    HAND_CLASS_PATH = 'classes'
    FILE_NAME_MODEL_ROCK_PAPER_SCISSORS = "rock-paper-scissors-model.h5"
    FILE_NAME_WEIGHT_ROCK_PAPER_SCISSORS = "rock-paper-scissors-weight-model.h5"
    FILE_NAME_MODEL = "hand-gesture-model.h5"
    FILE_WEIGHT_MODEL = "hand-gesture-weight-model.h5"
    VGG16_WIDTH, VGG16_HEIGHT = 224, 224
    IMAGE_CHANNELS = 3
    UNKNOWN_CLASS = "unknown"

    CLASS_MAP_ROCK_PAPER_SCISSORS = {
        "rock": 0,
        "paper": 1,
        "scissors": 2
    }

    CLASS_MAP_ITEMS = {
        "stop": 0,
        "close": 1,
        "ok": 2,
        "right": 3,
        "left": 4
    }

    CLASS_MAP_ITEMS_x = {
        "right": 3

    }

    def __init__(self):
        pass


# Deep Hand Gesture 
Hand gesture recognition using Python, OpenCV, Keras, Tensorflow, Matplotlib

## Getting Started
* All the source code is available inside SourceCode Directory. It requires python version 3.7 or later as to synchronize with tensorflow.
* The hand_recognize.py script will recognize the gesture according to the trained dataset.
* The hand_capture.py script file will help in creating your own dataset and hand_training_vgg16.py file 
  will use cnn deep neural nets to train your model and store it in the form of hadoop distributed (h5) format.
* Build the model with name "hand-gesture-model.h5" using hand_training_vgg16.py.
* Install the required libraries and packages.
* Start using the application by simply double clicking "hand_recognize.py"

# Deep Hand Gesture
https://www.youtube.com/watch?v=krFuaoRcBJo

An AI to run the DeepHandGesture application

## Requirements
Here is a short summary of the software and hardware requirements:
- OpenCV version 4.2.x 
- Anaconda Python 3 for installing Python and the required modules.
- Python version 3.7.x
- You can use any OS—macOS, Windows, and Linux-based OS—with this book.
- Its recommended you have at least 4 GB RAM in your system.
- It not necessary a GPU to run the code provided with these applications.

## Links for Anaconda Installers
https://www.anaconda.com/products/individual

Windows: https://repo.anaconda.com/archive/Anaconda3-2020.07-Windows-x86_64.exe

## Set up instructions
1. Clone the repo.
```sh
$ git clone https://github.com/HebertLuchettiRibeiro/deep_hand_gesture.git
$ cd deep_hand_gesture
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Capture the images for each gesture (rock, paper and scissors and None):
In this aplication, we will capture 250 training images,
50 validation images and 32 test images. All them for each hand gesture class
```sh
$ python hand_capture.py
```

4. Train the model
```sh
$ python hand_training_vgg16.py
```

5. Test the model on some images
```sh
$ python hand_predict.py
```

6. Run the hand class prediction
```sh
$ python hand_recognition.py
```

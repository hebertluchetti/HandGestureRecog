import numpy as np
import logging
import threading
import time
#import cv2
import logging
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
#from matplotlib import style
from threading import Thread

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

class BarHGraph:
    def __init__(self, predictData):
        plt.style.use('fivethirtyeight')
        self.predictData = predictData
        self.size = len(self.predictData.class_names)
        self.colors = plt.cm.Dark2(range(self.size))
        self.figure, self.ax = plt.subplots(tight_layout=True)
        self.loadBar()
        self.animation = FuncAnimation(self.figure, self.update, interval=1000, repeat=True)

    def update(self, i):
        return self.loadBar()

    def loadBar(self):
        self.ax.cla()
        self.bars = self.ax.barh(self.predictData.class_names, self.predictData.data, align='center', color=self.colors)
        self.ax.set_xlabel('Percentages', fontsize='medium')
        self.ax.set_title('Class predictions',fontsize='medium')
        self.niceAxes(self.ax)
        return self.bars

    def niceAxes(self, ax):
        ax.set_facecolor('.8')
        ax.tick_params(labelsize='medium', length=0)
        ax.grid(True, axis='x', color='white')
        ax.set_axisbelow(True)
        [spine.set_visible(False) for spine in ax.spines.values()]

    def show(self):
        plt.show()

class DataFetch(Thread):
    def __init__(self, predict_data):
        Thread.__init__(self,  daemon=True)
        self._predictData = predict_data
        self._period = 0.25
        self._nextCall = time.time()
        self._running = True

    def setPredictData(self, predict_data):
        self._predictData.data = predict_data.data

    def terminate(self):
        self._running = False

    # def run(self):
    #     while self._running:
    #         print("updating data=", self._predictData.data)
    #         # add data to data class
    #         self.size = len(self._predictData.class_names)
    #         self._predictData.data = 5 + 2 * np.random.rand(self.size)
    #         # sleep until next execution
    #         self._nextCall = self._nextCall + self._period
    #         time.sleep(self._nextCall - time.time())

class predictData():
    def __init__(self):
        self.class_names = ['stop', 'close', 'right', 'left', 'ok']
        self.data = [0, 0, 0, 0, 0]


# predData = predictData()
# plotter = BarHGraph(predData)
# fetcher = DataFetch(predData)
# fetcher.start()
# plotter.show()


#
# predData = predictData()
# plotter = BarHGraph(predData)
#fetcher = DataFetch(predData)

def thread_function(predictData):
    logging.info("Thread predictData starting :%s ", predictData.class_names)
    size = len(predictData.class_names)
    while True:
        print("updating data=", predictData.data)
        predictData.data = 5 + 2 * np.random.rand(size)
        time.sleep(1)

    logging.info("Thread [%s]: finishing", predictData.class_names)

if __name__ == "__main__":
    pred_data = predictData()
    plotter = BarHGraph(pred_data)

    logging.info("Main    : before creating thread")
    fetcher = threading.Thread(target=thread_function, args=(pred_data,),  daemon=True)
    #fetcher = DataFetch(pred_data)
    logging.info("Main    : before running thread")

    fetcher.start()
    plotter.show()

    logging.info("Main    : wait for the thread to finish")
    # x.join()
    logging.info("Main    : all done")

# def run(predictData):
#     fetcher = DataFetch(predData)
#     fetcher.start()
#     plotter.show()
#     while True:
#         print("updating data=", predictData.data)
#         # add data to data class
#         size = len(predictData.class_names)
#
#         predData.data = 5 + 2 * np.random.rand(size)
#
#         time.sleep(1)
#         #fetcher.join()
#
# run(predData)




#https://www.thetopsites.net/article/53952210.shtml
# class LiveBarHGraph:
#     def __init__(self, predictData, colors):
#         self.predictData = predictData
#         self.size = len(self.predictData.class_names)
#         self.colors = colors
#         self.figure, self.ax = plt.subplots(tight_layout=True)
#         self.loadBar()
#         self.animation = FuncAnimation(self.figure, self.update, interval=1000, repeat=True)
#         self.th = Thread(target=self.thread_f, daemon=True)
#         self.th.start()
#
#     def update(self, frame):
#         return self.loadBar()
#
#     def loadBar(self):
#         self.ax.cla()
#         self.bars = self.ax.barh(self.predictData.class_names, self.predictData.data, align='center', color=self.colors)
#         # self.ax.invert_yaxis()  # labels read top-to-bottom
#         self.ax.set_xlabel('Percentages', fontsize='medium')
#         self.ax.set_title('Class predictions',fontsize='medium')
#         self.niceAxes(self.ax)
#         return self.bars
#
#     def niceAxes(self, ax):
#         ax.set_facecolor('.8')
#         ax.tick_params(labelsize='medium', length=0)
#         ax.grid(True, axis='x', color='white')
#         ax.set_axisbelow(True)
#         [spine.set_visible(False) for spine in ax.spines.values()]
#
#     def show(self):
#         plt.show()
#
#     def setData(self, data):
#         self.predictData.data = data
#
#     def getData(self):
#         return self.predictData.data
#
#     def emitter(self, p=0.03):
#         'return a random value with probability p, else 0'
#         self.predictData.data = 5 + 2 * np.random.rand(self.size)
#
#     def thread_f(self):
#         while True:
#             self.predictData.data = 5 + 2 * np.random.rand(self.size)
#             print(self.predictData.data)
#             time.sleep(1)

# size = len(predData.class_names)
# colors = plt.cm.Dark2(range(size))
# plt.style.use('fivethirtyeight')
#
# g = LiveBarHGraph(predData, colors)
# g.show()

# while True:
#     data = 8 + 12 * np.random.rand(size)
#     g.setData(data)
#     time.sleep(1)
#     key = cv2.waitKey(10)
#     if key == 27:  # press 's' to capture the training and test images
#         break
"""

class LiveBarHGraph:
    def __init__(self, class_names, data, colors):
        self.class_names = class_names
        self.size = len(class_names)
        self.data = data
        self.colors = colors
        self.newData = data

        plt.style.use('fivethirtyeight')
        self.figure, self.ax = plt.subplots(tight_layout=True)
        self.loadBar()
        self.animation = FuncAnimation(self.figure, self.update, interval=1000)
        self.th = Thread(target=self.thread_f, daemon=True)
        self.th.start()

    def update(self, frame):
        return self.loadBar()

    def loadBar(self):
        self.ax.cla()
        self.bars = self.ax.barh(self.class_names, self.data, align='center', color=self.colors)
        # self.ax.invert_yaxis()  # labels read top-to-bottom
        self.ax.set_xlabel('Percentages', fontsize='medium')
        self.ax.set_title('Class predictions',fontsize='medium')
        self.niceAxes(self.ax)
        return self.bars

    def niceAxes(self, ax):
        ax.set_facecolor('.8')
        ax.tick_params(labelsize='medium', length=0)
        ax.grid(True, axis='x', color='white')
        ax.set_axisbelow(True)
        [spine.set_visible(False) for spine in ax.spines.values()]

    def show(self):
        plt.show()

    def setData(self, data):
        self.data = data

    def getData(self):
        return self.data

    def emitter(self, p=0.03):
        'return a random value with probability p, else 0'
        self.data = 5 + 2 * np.random.rand(size)

    def thread_f(self):
        while True:
            self.data = self.data = 5 + 2 * np.random.rand(size)
            print(self.data)
            time.sleep(1)

class MyDataClass():

    def __init__(self):
        self.class_names = ('stop', 'close', 'right', 'left', 'ok')
        self.data = (0, 0, 0, 0, 0)




class_names = ('stop', 'close', 'right', 'left', 'ok')
size = len(class_names)
data = 5 + 2 * np.random.rand(size)
colors = plt.cm.Dark2(range(size))

g = LiveBarHGraph(class_names, data, colors)
g.show()
"""

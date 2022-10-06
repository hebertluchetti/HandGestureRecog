from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class BarHGraph:
    def __init__(self, predictData):
        plt.style.use('fivethirtyeight')
        self.predictData = predictData
        self.size = len(self.predictData.class_names)
        self.colors = plt.cm.Dark2(range(self.size))
        self.figure, self.ax = plt.subplots(tight_layout=True)
        self.loadBar()
        self.animation = FuncAnimation(self.figure, self.update, interval=150)

    def update(self, i):
        return self.loadBar()

    def loadBar(self):
        self.ax.cla()
        self.bars = self.ax.barh(self.predictData.class_names, self.predictData.data, align='center', color=self.colors)
        self.ax.set_xlabel('Prediction (%)', fontsize='medium')
        self.ax.set_ylabel('Hand class', fontsize='medium')
        self.ax.set_title('Deep Hand Gesture Predictions', fontsize='medium')
        self.ax.invert_yaxis()  # labels read top-to-bottom
        self.niceAxes(self.ax)
        return self.bars

    def niceAxes(self, ax):
        ax.set_facecolor('.8')
        ax.tick_params(labelsize='medium', length=0)
        ax.grid(True, axis='x', color='white')
        ax.set_axisbelow(True)
        [spine.set_visible(False) for spine in ax.spines.values()]

    def show(self):
        plt.show(block=False)

class PredictData():
    def __init__(self, class_names):
        self.class_names = class_names
        self.default = [0, 0, 0, 0, 0]
        self.data = self.default

    def update(self, data):
        self.data = data
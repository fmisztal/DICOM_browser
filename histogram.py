from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
INFO = None


class HistogramDialog(QDialog):
    '''Class of histogram dialog window

    '''
    finished = Signal(np.ndarray)

    def __init__(self, file):
        '''Constructor of histogram dialog window

        :param file: dicom.pixel_array
        '''
        super().__init__()
        self.setWindowTitle("Histogram")
        self.resize(500, 550)
        self.file = file

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.scaleButton = QComboBox()
        self.scaleButton.addItems(['linear', 'log'])
        self.scaleButton.adjustSize()
        self.scaleButton.setCurrentIndex(0)
        self.scaleButton.currentIndexChanged.connect(self.drawPlot)

        self.infoLabel = QLabel()
        self.infoLabel.setText(INFO)
        self.infoLabel.adjustSize()

        self.paramLabel = QLabel()

        self.toolbarLayout = QHBoxLayout()
        self.toolbarLayout.addWidget(self.toolbar)
        self.toolbarLayout.addWidget(self.scaleButton)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.toolbarLayout)
        self.mainLayout.addWidget(self.canvas)
        self.mainLayout.addWidget(self.paramLabel)
        self.mainLayout.addWidget(self.infoLabel)

        self.setLayout(self.mainLayout)

        self.drawPlot()

    def drawPlot(self):
        """Draw histogram

        Function draws histogram in new window

        """
        self.figure.clear()
        ax = self.figure.add_subplot()

        imageData = self.file
        imgs_to_process = imageData.astype(np.float64)
        ax.hist(imgs_to_process.flatten(), bins=50, color='c')
        ax.set_xlabel("Hounsfield Units (HU)")
        ax.set_ylabel("Frequency")
        ax.set_yscale(self.scaleButton.currentText())

        self.canvas.draw()
        self.showParameters()

    def showParameters(self):
        """Show deviation and mean

        Function shows deviation and mean under histogram

        """
        self.paramLabel.setText('Deviation: ' + str(np.std(self.file)) +
                                '   Mean: ' + str(np.mean(self.file)))

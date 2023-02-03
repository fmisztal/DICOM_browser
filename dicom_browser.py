import histogram
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QPushButton, QWidget, QSlider, QGridLayout, QFileDialog, QMainWindow, \
    QComboBox, QVBoxLayout, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import pydicom
import glob
import sys
import cv2
import math
import shutil
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans

import roi
from filters import Kernels
from histogram import HistogramDialog
from roi import selectionWindow

matplotlib.use('Qt5Agg')


class Window(QMainWindow):
    """Class of main window

    This class holds all the program's main window objects and its functions

    """
    files = None
    currentFile = None
    slices = None
    currentSlice = None
    roiSlices = None
    currentRoiSlice = None
    isRoiActive = False
    newFile = True
    img_hu = None
    one_woksel = None
    dimensions = None
    draw = False
    firstClick = True
    lineStart = None
    lineStop = None

    def __init__(self, parent=None):
        """Constructor of main window class

        Initiates all components of the window

        """
        super(Window, self).__init__(parent)
        self.resize(700, 600)
        self.mainWidget = QWidget()
        self.buttonWidget = QWidget()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton()
        self.button.setText('Upload files')
        self.button.adjustSize()
        self.button.setMaximumWidth(200)
        self.button.clicked.connect(self.uploadFile)

        self.showAll = QPushButton()
        self.showAll.setText('Show all')
        self.showAll.adjustSize()
        self.showAll.setMaximumWidth(200)
        self.showAll.clicked.connect(self.plotAll)

        self.roiButton = QPushButton()
        self.roiButton.setText('Find lungs')
        self.roiButton.adjustSize()
        self.roiButton.setMaximumWidth(200)
        self.roiButton.clicked.connect(self.apply_lung_mask)

        self.wandButton = QPushButton()
        self.wandButton.setText('Apply ROI')
        self.wandButton.adjustSize()
        self.wandButton.setMaximumWidth(200)
        self.wandButton.clicked.connect(self.wand)

        self.colorButton = QComboBox()
        self.colors = [plt.cm.bone, plt.cm.twilight, plt.cm.plasma, plt.cm.inferno, plt.cm.magma, plt.cm.cividis]
        self.colorButton.addItems(['bone', 'twilight', 'plasma', 'inferno', 'magma', 'cividis'])
        self.colorButton.adjustSize()
        self.colorButton.setMaximumWidth(200)
        self.colorButton.setCurrentIndex(0)
        self.colorButton.currentIndexChanged.connect(self.plot)

        self.hu_info = QLabel()
        self.hu_info.setText("HU = 0")
        self.hu_info.adjustSize()

        self.length = QLabel()
        self.length.setText("Length: 0mm")
        self.length.adjustSize()

        self.thresholdSlider = QSlider()
        self.thresholdSlider.valueChanged[int].connect(self.setTresholdValue)
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setEnabled(False)

        self.thresholdValue = QLabel()
        self.thresholdValue.adjustSize()

        self.filters = Kernels()
        self.filters.setMaximumWidth(200)
        self.filters.currentIndexChanged.connect(self.plot)

        self.histButton = QPushButton()
        self.histButton.setText('Histogram')
        self.histButton.adjustSize()
        self.histButton.setMaximumWidth(200)
        self.histButton.clicked.connect(self.histogram)

        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.button)
        buttonLayout.addWidget(self.histButton)
        buttonLayout.addWidget(self.roiButton)
        buttonLayout.addWidget(self.showAll)
        buttonLayout.addWidget(self.colorButton)
        buttonLayout.addWidget(self.filters)
        buttonLayout.addWidget(self.wandButton)
        buttonLayout.addWidget(self.thresholdSlider)
        buttonLayout.addWidget(self.thresholdValue)
        buttonLayout.addStretch()
        self.buttonWidget.setLayout(buttonLayout)

        self.slider = QSlider()
        self.slider.valueChanged[int].connect(self.changeFrame)
        self.slider.setDisabled(True)

        self.frameLabel = QLabel()
        self.frameLabel.setText('0')
        self.frameLabel.adjustSize()

        frameSliderLayout = QVBoxLayout()
        frameSliderLayout.addWidget(self.frameLabel)
        frameSliderLayout.addWidget(self.slider)

        self.imgRangeSlider1 = QSlider()
        self.imgRangeSlider1.valueChanged[int].connect(self.plot)
        self.imgRangeSlider1.setOrientation(QtCore.Qt.Horizontal)
        self.imgRangeSlider1.setSliderPosition(0)
        self.imgRangeSlider1.setEnabled(False)

        self.rangeLabel1 = QLabel()
        self.rangeLabel1.setText('V1 = 0')
        self.rangeLabel1.adjustSize()

        self.imgRangeSlider2 = QSlider()
        self.imgRangeSlider2.valueChanged[int].connect(self.plot)
        self.imgRangeSlider2.setOrientation(QtCore.Qt.Horizontal)
        self.imgRangeSlider2.setSliderPosition(0)
        self.imgRangeSlider2.setEnabled(False)

        self.rangeLabel2 = QLabel()
        self.rangeLabel2.setText('V2 = 0')
        self.rangeLabel2.adjustSize()

        range1Layout = QHBoxLayout()
        range1Layout.addWidget(self.rangeLabel1)
        range1Layout.addWidget(self.imgRangeSlider1)

        range2Layout = QHBoxLayout()
        range2Layout.addWidget(self.rangeLabel2)
        range2Layout.addWidget(self.imgRangeSlider2)

        layout = QGridLayout()
        layout.addWidget(self.toolbar, 0, 1)
        layout.addWidget(self.canvas, 1, 1)
        layout.addLayout(range1Layout, 2, 1)
        layout.addLayout(range2Layout, 3, 1)
        layout.addWidget(self.buttonWidget, 1, 0)
        layout.addLayout(frameSliderLayout, 1, 2)
        layout.addWidget(self.hu_info, 3, 0)
        layout.addWidget(self.length, 2, 0)
        self.mainWidget.setLayout(layout)
        self.setCentralWidget(self.mainWidget)

    def wand(self):
        '''Opens up Roi dialog window

        It runs the selectionWindow dialog from the roi.py,
        passing currently displayed dicom path as an argument.
        After that it deletes cache folder.

        '''
        print(self.currentFile)
        roi = selectionWindow(self.currentFile)
        roi.exec()
        shutil.rmtree('.\\cache\\')

    def setTresholdValue(self, value):
        '''self.thresholdSlider action support

        Function setting up self.thresholdValue label's text to display current state of slider,
        switches off the possibility of drawing and refreshes the canvas.

        :param value: current value of a self.thresholdSlider
        '''
        self.thresholdValue.setText('Threshold: ' + str(value))
        self.draw = False
        self.plot()

    def histogram(self):
        '''Runs the histogram dialog

        If there is a self.currentSlice, it runs the HistogramDialog dialog from
        histogram.py based on self.image passed as parameter.

        '''
        if self.currentSlice != None:
            dlg = HistogramDialog(self.image)
            dlg.exec()

    def apply_lung_mask(self):
        '''Function to enable/disable lung mask mode

        Function switching on/off the lung mask mode and accordingly setting up the environment.

        '''
        if not self.slices:
            return
        if self.roiButton.text() == 'Disable mask':
            self.roiButton.setText('Find lungs')
            self.hu_info.setText('HU = ')
            self.isRoiActive = False
            self.slider.setMaximum(len(self.files) - 1)
            self.currentFile = self.files[0]
            self.currentSlice = self.slices[0]
            self.slider.setValue(0)
            self.thresholdSlider.setEnabled(True)
            self.draw = False
            self.plot()
            return
        imgs_to_process = self.one_woksel

        self.masked_lung = []
        for img in imgs_to_process:
            self.masked_lung.append(self.make_lungmask(img))
        self.roiButton.setText('Disable mask')
        self.hu_info.setText('')
        self.isRoiActive = True
        self.slider.setMaximum(len(self.masked_lung) - 1)
        self.slider.setValue(0)
        self.thresholdSlider.setEnabled(False)
        self.draw = False
        self.plot()

    def make_lungmask(self, img):
        '''Applies lung mask on all slices

        :param img:
        '''
        row_size = img.shape[0]
        col_size = img.shape[1]

        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std

        middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)

        img[img == max] = mean
        img[img == min] = mean

        kmeans = KMeans(n_clusters=2, n_init=10).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)

        eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
        dilation = morphology.dilation(eroded, np.ones([8, 8]))

        labels = measure.label(dilation)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
                2] < col_size / 5 * 4:
                good_labels.append(prop.label)
        mask = np.ndarray([row_size, col_size], dtype=np.int8)
        mask[:] = 0

        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))
        return mask * img

    def load_scan(self):
        '''Prepares dicom files for use

        This function sorts all dicom files based on their InstanceNumber
        and sets up its thickness if that's possible.

        '''
        slices = [pydicom.dcmread(s) for s in self.files]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        try:
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            for s in slices:
                s.SliceThickness = slice_thickness
        except:
            pass

        self.slices = slices

    def setThresholdSlider(self):
        '''Sets up thresholdSlider state

        '''
        image = self.get_pixels_hu(self.currentSlice)
        self.thresholdSlider.setMinimum(image.min())
        self.thresholdSlider.setMaximum(image.max())
        self.thresholdSlider.setSliderPosition(self.thresholdSlider.minimum())

    def sort(self):
        """Sort dicom files

        Function sorts dicom files in directory

        """
        self.files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)

    def uploadFile(self):
        '''Function used to upload dicom files from chosen folder

        Function that opens up the QFileDialog window allowing to choose folder with dicom files to upload.
        It that ends successfully, it turns on some components and loads info to appropriate variables.

        '''
        directory = QFileDialog.getExistingDirectory(None, "QFileDialog.getOpenFileName()", "")
        if directory:
            self.files = glob.glob(directory + "/*.dcm")
            if len(self.files) < 1:
                self.slider.setSliderPosition(0)
                self.slider.setDisabled(True)
                return
            self.slider.setEnabled(True)
            self.slider.setMaximum(len(self.files) - 1)
            self.sort()
            self.currentFile = self.files[0]
            self.load_scan()
            self.currentSlice = self.slices[0]
            self.one_woksel = np.stack([s.pixel_array for s in self.slices])
            self.dimensions = [self.currentSlice.SliceThickness * self.one_woksel.shape[0],
                               self.currentSlice.PixelSpacing[0] * self.one_woksel.shape[1],
                               self.currentSlice.PixelSpacing[1] * self.one_woksel.shape[2]]
            histogram.INFO = f"Overall thickness: {self.dimensions[0]} mm   Width: {self.dimensions[1]} mm" \
                             f"   Height: {self.dimensions[2]} mm"
            self.newFile = True
            self.thresholdSlider.setEnabled(True)
            self.imgRangeSlider1.setEnabled(True)
            self.imgRangeSlider2.setEnabled(True)
            self.setThresholdSlider()
            self.draw = False
            self.plot()

    def changeFrame(self, value):
        '''Loads new slice on canvas

        Function that loads a new file and slice based on self.slider value and refreshing the canvas.

        :param value: current value of a self.slider
        '''
        if self.files:
            self.currentFile = self.files[value]
            self.currentSlice = self.slices[value]
            self.frameLabel.setText(str(value))
            self.setThresholdSlider()
            self.plot()

    def get_pixels_hu(self, dcm):
        """Convert image values to Hu values and deleting background

        Function deletes background values and normalize values in Hu scale

        :param dcm: dicom picture
        :return np.array numpy array with Hu values
        """
        image = dcm.pixel_array
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def setNorm(self):
        '''Sets norm of display

        Function that is used to apply contrast by switching the view field between
        values selected on imgRangeSlider1 and imgRangeSlider2.

        :return: norm
        '''
        val1 = self.imgRangeSlider1.value()
        val2 = self.imgRangeSlider2.value()
        norm = None
        if val1 != val2:
            norm = plt.Normalize(val1, val2, True)
        return norm

    def plotAll(self):
        '''Opens up figure window with all slices

        Function that displays all slices with currently applied
        filters, color palettes and lung mask if switched on.

        :return:
        '''
        if not self.slices:
            return
        if self.isRoiActive:
            slices = self.masked_lung
            images = slices
        else:
            slices = self.slices
            images = np.stack([s.pixel_array for s in slices])
            for i in range(len(images)):
                images[i] = self.get_pixels_hu(self.slices[i])

        numOfSlices = len(slices)
        rows = int(((numOfSlices ** 0.5) ** 2) ** 0.5)
        lol = divmod(numOfSlices - (rows ** 2), rows)
        columns = rows + lol[0]
        if lol[1] != 0:
            columns += 1

        fig, ax = plt.subplots(rows, columns, figsize=[12, 12])
        x, y = 0, 0
        img_norm = self.setNorm()
        for i in range(rows * columns):
            if i < len(slices):
                ax[x, y].set_title('slice %d' % i)
                images[i] = cv2.filter2D(images[i], -1, self.filters.getKernel())
                ax[x, y].imshow(images[i], cmap=self.colors[self.colorButton.currentIndex()], norm=img_norm)
            ax[x, y].axis('off')
            y += 1
            if y == columns:
                x += 1
                y = 0
        fig.show()

    def pixelSpacing(self):
        """Getting pixel spacing

        Function reads pixel spacing from dcm tags

        :return:
        """
        if self.currentSlice:
            dcm = self.currentSlice
            if hasattr(dcm, 'PixelSpacing'):
                roi.SPACING = dcm.PixelSpacing

    def setSlilders(self):
        '''Sets up sliders state

        '''
        self.imgRangeSlider1.setMinimum(self.image.min())
        self.imgRangeSlider1.setMaximum(self.image.max())

        self.imgRangeSlider2.setMinimum(self.image.min())
        self.imgRangeSlider2.setMaximum(self.image.max())
        if self.imgRangeSlider1.value() > self.imgRangeSlider2.value():
            self.imgRangeSlider1.setValue(self.imgRangeSlider2.value())
        self.rangeLabel1.setText('V1 = ' + str(self.imgRangeSlider1.value()))
        self.rangeLabel2.setText('V2 = ' + str(self.imgRangeSlider2.value()))

    def plot(self):
        '''Displays current slice on canvas

        Function that applies all chosen effects on chosen image and then drawing it on canvas

        '''
        if not self.files:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.isRoiActive:
            self.length.setText("Length: 0mm")
            image = self.masked_lung[self.slider.value()]
        else:
            image = self.get_pixels_hu(self.currentSlice)

        self.pixelSpacing()

        image = cv2.filter2D(image, -1, self.filters.getKernel())

        if self.thresholdSlider.value() != self.thresholdSlider.minimum() and not self.isRoiActive:
            self.length.setText("Length: 0mm")
            image = np.where(image < self.thresholdSlider.value(), 1.0, 0.0)

        self.image = image

        self.setSlilders()

        img_norm = self.setNorm()

        if self.draw and not self.isRoiActive and self.thresholdSlider.value() == self.thresholdSlider.minimum():
            image = cv2.line(image, self.lineStart, self.lineStop, color=(2000, 2000, 2000), thickness=4)
            l = round(float((math.sqrt(((self.lineStart[0] - self.lineStop[0]) ** 2) * roi.SPACING[0]
                                       + ((self.lineStart[1] - self.lineStop[1]) ** 2) * roi.SPACING[1]))), 2)
            self.length.setText(f"Length: {l} mm")

        ax.imshow(image, cmap=self.colors[self.colorButton.currentIndex()], norm=img_norm)
        self.canvas.draw()

    def on_move(self, event):
        """Reading mouse coordinate and dispaly Hu value

        Function reads mouse coordinate and display Hu values in this point

        :param event:
        """
        if self.files and (event.xdata or event.ydata) and not self.isRoiActive:

            self.img_hu = self.image
            self.newFile = False
            self.hu_info.setText(f"HU = {self.img_hu[int(event.ydata)][int(event.xdata)]}")
            print(int(event.ydata), int(event.xdata))

    def on_press(self, event):
        """Getting mouse press coordinate

        Function set coordinate of first left, second left and right click and change parameters to draw line

        :param event:
        """
        if event.xdata != None and event.ydata != None:
            if event.button == 1:
                if self.firstClick:
                    self.draw = False
                    self.length.setText("Length: 0mm")
                    self.plot()
                    self.lineStart = (int(event.xdata), int(event.ydata))
                    self.firstClick = False
                else:
                    self.lineStop = (int(event.xdata), int(event.ydata))
                    self.draw = True
                    self.plot()
                    self.firstClick = True

            elif event.button == 3:
                self.draw = False
                self.length.setText("Length: 0mm")
                self.plot()
                self.firstClick = True


if __name__ == '__main__':
    '''Main function
    
    '''
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec())

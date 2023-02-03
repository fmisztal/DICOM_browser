from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QSlider, QVBoxLayout, QLabel, QHBoxLayout, QDialog, QPushButton
from PySide6.QtGui import QPixmap, QImage
import pydicom
import cv2
import numpy as np
import dicom2jpg
import os
import math

SPACING = None


class ImageLabel(QLabel):
    '''Class of a label displaying image in selectionWindow class

    '''

    def __init__(self, parent):
        '''Constructor of ImageLabel

        Sets up self.parent as a parameter parent (selectionWindow class)

        '''
        super().__init__(parent)
        self.parent = parent

    def mousePressEvent(self, event):
        """Gets mouse press coordinates

        Function gets mouse press coordinates and calculate them on pixmap

        :param event:
        """
        self.x = event.position().x()
        self.y = event.position().y()
        pixmap_pos = self.pos() + QPoint(
            (self.width() - self.pixmap().width()) / 2,
            (self.height() - self.pixmap().height()) / 2
        ) - QPoint(10, 10)
        if (
                self.x >= pixmap_pos.x() and
                self.x < pixmap_pos.x() + self.pixmap().width() and
                self.y >= pixmap_pos.y() and
                self.y < pixmap_pos.y() + self.pixmap().height()
        ):
            x = event.position().x() - pixmap_pos.x()
            y = event.position().y() - pixmap_pos.y()

            if self.parent.isDrawActive:
                if event.button() == Qt.LeftButton:
                    if self.parent.firstClick:
                        self.parent.canDraw = False
                        self.parent.drawLine()
                        self.parent.lineStart = (int(x), int(y))
                        self.parent.firstClick = False
                    else:
                        self.parent.lineStop = (int(x), int(y))
                        self.parent.canDraw = True
                        self.parent.drawLine()
                        self.parent.firstClick = True
                else:
                    self.parent.canDraw = False
                    self.parent.firstClick = True
                    self.parent.drawLine()
            else:
                if event.button() == Qt.LeftButton:
                    ALT = False
                    SHIFT = False
                    if event.modifiers() & Qt.AltModifier:
                        ALT = True
                    if event.modifiers() & Qt.ShiftModifier:
                        SHIFT = True
                    self.parent.ev(x, y, ALT, SHIFT)


def _find_exterior_contours(img):
    '''Function finding contours

    :param img: image
    :return: contours
    '''
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


def search_folder(folder):
    """Search directory to find jpg

    :param folder: directory to search
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                return img


class selectionWindow(QDialog):
    '''Class of ROI dialog window

    '''
    isDrawActive = False
    firstClick = True
    lineStart = None
    lineStop = None
    canDraw = False

    def __init__(self, file, parent=None):
        '''Constructor of ROI dialog window

        :param file: dicom file path
        :param parent: QDialog
        '''
        super(selectionWindow, self).__init__(parent)
        self.resize(700, 600)

        self.dcm = pydicom.dcmread(file)
        dicom2jpg.dicom2jpg(file, target_root='.\\cache\\', anonymous=False)
        self.img = search_folder('.\\cache\\')
        self.img_with_con = self.img.copy()

        can_img = QImage(self.img.data, self.img.shape[1], self.img.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(can_img)

        self.connectivity = 4
        self.tolerance = 32
        h, w = self.img.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (self.connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)
        self.tolerance = (self.tolerance,) * 3

        self.canvas = ImageLabel(self)
        self.canvas.setPixmap(self.pixmap)
        self.canvas.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.adjustSize()

        self.drawButton = QPushButton()
        self.drawButton.setText('Draw')
        self.drawButton.adjustSize()
        self.drawButton.clicked.connect(self.activeDraw)

        self.length = QLabel()
        self.length.setText("Length: 0mm")
        self.length.adjustSize()

        self.surface = QLabel()
        self.surface.setText("Surface area: ")
        self.surface.adjustSize()

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.valueChanged[int].connect(self.change_tolerance)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setSliderPosition(32)
        self.label.setText('Tolerance: ' + str(self.slider.value()))

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.drawButton)
        slider_layout.addWidget(self.surface)
        slider_layout.addWidget(self.length)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.canvas)
        layout.addLayout(slider_layout)
        self.setLayout(layout)

    def activeDraw(self):
        """Set the draw button and logic variables after press

        """
        if not self.isDrawActive:
            self.isDrawActive = True
            self.drawButton.setText("Quit draw")
        else:
            self.isDrawActive = False
            self.firstClick = True
            self.drawButton.setText("Draw")

    def change_tolerance(self, value):
        """Change wand tolerance

        :param value: slider value
        """
        self.tolerance = (value,) * 3
        self.label.setText('Tolerance: ' + str(self.slider.value()))

    def ev(self, x, y, ALT, SHIFT):
        """Set the wand mask and call draw function

        :param x: x coordinate
        :param y: y coordinate
        :param ALT: flag if ALT was pressed
        :param SHIFT: flag if SHIFT was pressed
        """
        self._flood_mask[:] = 0
        x = int(x)
        y = int(y)
        cv2.floodFill(
            self.img,
            self._flood_mask,
            (x, y),
            (0, 0, 255),
            self.tolerance,
            self.tolerance,
            self._flood_fill_flags,
        )
        flood_mask = self._flood_mask[1:-1, 1:-1].copy()

        if ALT and SHIFT:
            self.mask = cv2.bitwise_and(self.mask, flood_mask)
        elif SHIFT:
            self.mask = cv2.bitwise_or(self.mask, flood_mask)
        elif ALT:
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(flood_mask))
        else:
            self.mask = flood_mask

        self.draw()

    def draw(self):
        """Set the roi and draw line if there was one before

        Function applies wand mask and check if line was drawn on image.
        Function checks surface area and calclate it to mm.

        """
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv2.drawContours(viz, contours, -1, color=(0, 0, 255), thickness=-1)
        viz = cv2.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=(0, 0, 255), thickness=1)
        self.img_with_con = viz.copy()
        if self.canDraw:
            cv2.line(viz, self.lineStart, self.lineStop, color=(255, 0, 0), thickness=4)
            l = round(float((math.sqrt(((self.lineStart[0] - self.lineStop[0]) ** 2) * SPACING[0] + (
                    (self.lineStart[1] - self.lineStop[1]) ** 2) * SPACING[1]))), 2)
            self.length.setText(f"Length: {l}mm")
        counter = 0
        for x in range(viz.shape[0]):
            for y in range(viz.shape[1]):
                if viz[x][y][1] != viz[x][y][2]:
                    counter += 1
        self.surface.setText(f"Surface area: {round(SPACING[0] * SPACING[1] * counter, 2)} mm^2")

        can_img = QImage(viz.data, viz.shape[1], viz.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(can_img)
        self.canvas.setPixmap(self.pixmap)

    def drawLine(self):
        """Draw line or delete it

        Function will draw line after 2 left-clicks in draw mode or erase it after right-click or third left-click

        """
        vizl = self.img_with_con.copy()
        self.length.setText("Length: 0mm")
        if self.isDrawActive and self.canDraw:
            cv2.line(vizl, self.lineStart, self.lineStop, color=(255, 0, 0), thickness=4)
            l = round(float(math.sqrt(((self.lineStart[0] - self.lineStop[0]) ** 2) * SPACING[0] + (
                    (self.lineStart[1] - self.lineStop[1]) ** 2) * SPACING[0])), 2)
            self.length.setText(
                f"Length: {l}mm")
        can_img = QImage(vizl.data, vizl.shape[1], vizl.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(can_img)
        self.canvas.setPixmap(self.pixmap)

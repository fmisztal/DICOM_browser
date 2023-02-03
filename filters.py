from PySide6.QtWidgets import QComboBox
import numpy as np


class Kernels(QComboBox):
    '''Class of QComboBox container containing all the matrix filters

    '''
    # Normal
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    # Edge Detection2
    kernel2 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    # Bottom Sobel Filter
    kernel3 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # Top Sobel Filter
    kernel4 = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    # Left Sobel Filter
    kernel5 = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    # Right Sobel Filter
    kernel6 = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    # Sharpen
    kernel7 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    # Emboss
    kernel8 = np.array([[-2, -1, 0],
                        [-1, 1, 1],
                        [0, 1, 2]])
    # Box Blur
    kernel9 = (1 / 9.0) * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
    # Gaussian Blur 3x3
    kernel10 = (1 / 16.0) * np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 1]])
    # Gaussian Blur 5x5
    kernel11 = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
    # Unsharp masking 5x5
    kernel12 = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                        [4, 16, 24, 16, 4],
                                        [6, 24, -476, 24, 6],
                                        [4, 16, 24, 16, 4],
                                        [1, 4, 6, 4, 1]])
    # Horizontal Prewitt
    kernel13 = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])

    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9,
               kernel10, kernel11, kernel12, kernel13]

    def __init__(self):
        '''Constructor of class with kernels

        '''
        super().__init__()
        self.addItems(["Normal", "Edge", "Bottom Sobel", "Top Sobel", "Left Sobel", "Right sobel",
                       "Sharpen", "Emboss", "Blur", "Gaussian Blur 3x3", "Gaussina Blur 5x5",
                       "Unsharpmask 5x5", "Horizontal Prewitt"])
        self.setCurrentIndex(0)

    def getKernel(self):
        '''Function returning filter

        :return: currently selected filter
        '''
        return self.kernels[self.currentIndex()]

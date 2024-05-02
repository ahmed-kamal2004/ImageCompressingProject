import cv2 as cv
from utilites import Simulator,OutputController
from scipy.fftpack import dct,idct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
import cv2
import sys
import numpy
import PIL
from PIL import Image


if __name__ == "__main__":

    image = cv2.imread("image1.png")

    psnrArr = np.zeros((4))
    for i in range(1,5):
        original,compressed,decompressed = Simulator.startSim(image,i,8)
        psnrArr[i-1] = OutputController.mainController(original,compressed,decompressed,i)
    OutputController.plotter(psnrArr)


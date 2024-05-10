from utilites import Simulator,OutputController
import numpy as np
import cv2


if __name__ == "__main__":

    image = cv2.imread("image1.png")

    psnrArr = np.zeros((4))
    originalArr = np.zeros((4))
    compressedArr = np.zeros((4))
    decompressedArr = np.zeros((4))
    for i in range(1,5):
        original,compressed,decompressed = Simulator.startSim(image,i,8)

        psnrArr[i-1],originalArr[i-1],compressedArr[i-1],decompressedArr[i-1]= OutputController.mainController(original,compressed,decompressed,i)
    OutputController.plotter(psnrArr)

    print(originalArr)
    print(compressedArr)
    print(decompressedArr)
    print(psnrArr)


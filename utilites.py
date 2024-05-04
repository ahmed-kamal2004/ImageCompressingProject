import numpy as np
import cv2
from scipy.fftpack import dct, idct , dctn, idctn
import matplotlib.pyplot as plt
from math import log10
import matplotlib.pyplot as plt


class Simulator:
    @staticmethod 
    def startSim(img,m,blockSize):

        sizeRequired = img.shape
        ## Compressor
        compressedArr = BlkOperations.compressorGen(img,m,blockSize)

        ## Decompressor
        deComprossedArr = BlkOperations.decompressorGen(compressedArr,m,sizeRequired,blockSize)

        return img,compressedArr,deComprossedArr




class BlkOperations:
    @staticmethod
    def decompressorGen(compressedMatrix,m,sizeReuqired,blocksize = 8):
        ## Decompressing the Image

        verticalSize = sizeReuqired[0]//blocksize
        horizontalSize = sizeReuqired[1]//blocksize
        outputMatrix = np.zeros(sizeReuqired)
        for i in range(0,verticalSize):
            for j in range(0,horizontalSize):
                matrix = np.zeros((blocksize,blocksize))
                x = i * blocksize
                y = j * blocksize
                for k in range(sizeReuqired[2]):
                    matrix[0:m,0:m] = (compressedMatrix[:,:,k])[i*m : (i+1) * m , j*m : (j+1) * m]
                    outputMatrix[:,:,k][x:x+blocksize,y:y+blocksize] = BlkOperations.deCompress(matrix)
        return outputMatrix

    @staticmethod
    def compressorGen(OriginalMatrix,m,blocksize = 8):
        ## Compressing The Image 

        verticalSize = OriginalMatrix.shape[0]//blocksize
        horizontalSize = OriginalMatrix.shape[1]//blocksize
        outputMatrix = np.zeros((verticalSize*m,horizontalSize*m,OriginalMatrix.shape[2]))
        for i in range(0,verticalSize):
            for j in range(0,horizontalSize):
                x = i * blocksize
                y = j * blocksize
                for k in range(OriginalMatrix.shape[2]):
                    matrix = OriginalMatrix[:,:,k][x:x+blocksize,y:y+blocksize]
                    outputMatrix[:,:,k][i*m : (i+1) * m , j*m : (j+1) * m] = BlkOperations.enCompress(matrix,m)
        return outputMatrix


    @staticmethod
    def deCompress(matrix):
        """
        deCompress function applys Inverse 2d Discrete Cosine Transform (DCT) algoritm on a given square matrix
        param1 : 2d ndarray of shape (x,x)
        returns : De-compressed 2d ndarray of shape (x,x)
        """
        # return idct(idct(matrix.T,norm = "ortho").T,norm = "ortho")
        return idctn(matrix , norm = "ortho")

    @staticmethod
    def enCompress(matrix,m): 
        """
        enCompress function applys 2d Discrete Cosine Transform (DCT) algoritm on a given square matrix
        param1 : 2d ndarray 
        param2 : integer
        returns : Compressed 2d ndarray of shape (m,m) 
        """
        # return dct(dct(matrix.T,norm = "ortho").T,norm = "ortho")[0:m,0:m]
        return dctn(matrix , norm = "ortho")[:m,:m]


class OutputController:

    @staticmethod
    def mainController(original,compressed,decompressed,m):
        print(f""" At m = {m}\nOriginal Matrix size is {original.shape[0]} {original.shape[1]} {original.shape[2]}\n
Compressed Matrix size is {compressed.shape[0]} {compressed.shape[1]} {compressed.shape[2]}\n
Decompressed Matrix size is {decompressed.shape[0]} {decompressed.shape[1]} {decompressed.shape[2]}\n
              """)
        PSNR = OutputController.psnrCalc(original,decompressed)
        print(f" PSNR Rate is {PSNR}")

        cv2.imwrite(f"Compressed_{m}_.jpg",compressed)
        cv2.imwrite(f"DeCompressed_{m}_.jpg",decompressed)

        return PSNR
        
    

    def mseCalc(original,decompressed):
        return (1/(original.shape[0] * original.shape[1]))*((original - decompressed)**2).sum()

        pass
    def psnrCalc(original,decompressed):
        return 10 * log10((255*255)/OutputController.mseCalc(original,decompressed))
    
    def plotter(arr):
        
        plt.plot(arr, np.arange(1, 5), 'bo')
        plt.savefig("PSNR_With_Time.png")
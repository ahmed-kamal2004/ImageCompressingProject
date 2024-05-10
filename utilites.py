import numpy as np
import cv2
from scipy.fftpack import dct, idct , dctn, idctn
import matplotlib.pyplot as plt
from math import log10
import sys
import os
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
        PSNR = OutputController.psnrCalc(original,decompressed)
        cv2.imwrite(f"Compressed_{m}_.png",compressed)
        cv2.imwrite(f"DeCompressed_{m}_.png",decompressed)
        orig = os.path.getsize("image1.png")/(1024**2)
        comp = os.path.getsize(f"Compressed_{m}_.png")/(1024**2)
        decomp = os.path.getsize(f"DeCompressed_{m}_.png")/(1024**2)
        return PSNR,orig,comp,decomp
        
    
    def mseCalc(original,decompressed):
        return (1/(original.shape[0] * original.shape[1] * original.shape[2]))*((original - decompressed)**2).sum()
    def psnrCalc(original,decompressed):
        return 10 * log10((255*255)/OutputController.mseCalc(original,decompressed))
    def plotter(arr):
        plt.plot( np.arange(1, 5),arr, 'bo')
        plt.savefig("PSNR_With_Time.png")
import cv2 as cv
import numpy as np

#np.set_printoptions(linewidth=np.inf, threshold = np.nan)
#resize img defined InterpolationSize
def Interpolation(img,InterpolationSize):
    img = cv.resize(img, (0,0), fx = InterpolationSize, fy = InterpolationSize, interpolation = cv.INTER_LINEAR)
    return img

#identify Neurons
def IdentifyNeurons(path, threshold = 100, areamin = 40, areamax = 400, InterpolationSize = 1,):
    img = cv.imread(path,-1)
    img = Interpolation(img, InterpolationSize)
    blur = cv.GaussianBlur(img,(5,5),0)
    #kernel = np.ones((5,5),np.uint8)
    #tophat = cv.morphologyEx(blur, cv.MORPH_TOPHAT, kernel)
    tophat = blur
    NormalizedImg = np.zeros(tophat.shape)
    NormalizedImg = cv.normalize(tophat,NormalizedImg, 0, 255, cv.NORM_MINMAX) 
    ret3,th3 = cv.threshold(np.uint8(NormalizedImg), threshold, 255, cv.THRESH_BINARY)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(th3)
    #stats[i] is [x0, y0, width, height, area]
    #cv.imwrite('outpath',th3)
    neurons = []
    for i in range(1,len(stats)):
        if stats[i][4]>areamin and stats[i][4]<areamax:
            neurons.append(stats[i])
    return neurons

def NeuronLikelyhood():
    return False

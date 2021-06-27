import sys
import cv2 as cv
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt

def histograma(modelo):
    numLabels = np.arange(0,len(np.unique(modelo.labels_)) + 1)
    (hist,_) = np.histogram(modelo.labels_,bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def sacarColores(hist, centroides):
    bar = np.zeros((50,300,3),dtype="uint8")

    startX = 0

    for (percent, color) in zip(hist, centroides):         
        #if not esVerde(color):
        #    break
        print("color=",color)
        print("percent=",percent)
        endX = startX + (percent * 300)
        cv.rectangle(bar,(int(startX),0),(int(endX),60),color.astype("uint8").tolist(),-1)       
        startX = endX
            
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
                          
    return color

def esVerde(color):
    r,g,b = color
    if not (r >= 0 and r <= 144):
        return False
    if not( g >= 100 and g <= 238):
        return False
    if not ( b >=0 and b <=144):
        return False
    return True

def sacarColorUniforme(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)   

    img = img.reshape((img.shape[0]*img.shape[1],3))    

    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(img)

    hist = histograma(kmeans)
    colorUniforme = sacarColores(hist,kmeans.cluster_centers_)

    print(colorUniforme)

def main(argv):
    img = cv.imread(argv[0])
    sacarColorUniforme(img)              

if __name__ == "__main__":
    main(sys.argv[1:])
import sys
import json
import numpy as np
import joblib
from sklearn import ensemble, metrics,cluster

#leer las clases de cada foto
def leerJSON(nombreArchivo):
    clases = []
    with open(nombreArchivo, 'r') as file:
        jsonData = file.read()
        etiquetas = json.loads(jsonData)
        for etiqueta in etiquetas:
            clases.append(etiquetas[etiqueta]["denominacion"])
    return clases

#calcula las estadisticas de cada clase (precision, recall, f1-score)
def estadisticasPorClase(y_prueba, y_pred):

    matrizConfusion = metrics.confusion_matrix(y_prueba,y_pred)
    
    nuevaMatriz = [[0,0],[0,0]]
    clase = "1"
    for i in range(8):
        pos11 = matrizConfusion[i][i]
        pos00 = matrizConfusion[0][0] + matrizConfusion[1][1] + matrizConfusion[2][2] + matrizConfusion[3][3] - pos11                
        pos01 = 0
        pos10 = 0
        for j in range(8):
            pos10 += matrizConfusion[i][j]
            pos01 += matrizConfusion[j][i]
        pos01 -= pos11
        pos10 -= pos11
        nuevaMatriz[0][0] = pos00
        nuevaMatriz[0][1] = pos01
        nuevaMatriz[1][0] = pos10
        nuevaMatriz[1][1] = pos11

        recall = 0
        precision = 0
        f1 = 0
        if (pos11 + pos10) != 0:
            recall = (pos11 / (pos11 + pos10)) * 100    
        if (pos11 + pos01) != 0:
            precision = (pos11 / (pos11 + pos01)) * 100
        if (precision + recall) != 0:
            f1 = round((2 * precision * recall) / (precision + recall), 5)
        
        if i == 1:
            clase = "2"
        elif i == 2:
            clase = "5"
        elif i == 3:
            clase = "10"
        elif i == 4:
            clase = "20"
        elif i == 5:
            clase = "50"
        elif i == 6:
            clase = "100"
        elif i == 7:
            clase = "500"


        print()
        print("--------------------------------------------")
        print(clase)
        imprimirMatrizConfusion(clase,nuevaMatriz)
        print("Recall:  %s" % (round(recall,5)))
        print("Precision:  %s" % (round(precision,5)))
        print("F1-Score:  %s" % (f1))
    
    print()
    print("--------------------------------------------")
    #recall promedio
    print("Recall Promedio: ",round(metrics.recall_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))
    #preciosion promedio
    print("Precision Promedio: ",round(metrics.precision_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))
    #f1-socre promedio
    print("F1-Score Promedio: ",round(metrics.f1_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))

def imprimirMatrizConfusion(clase, matriz):
    print("             Sobrantes   %s" % (clase))
    cont = 0
    for fila in matriz:
        if cont == 0:
            print("Sobrantes       ",end="")
        else:
            print("%s          " % (clase),end="")
        for columna in fila:
            print("%s         " % (columna),end="")
        print("")
        cont+=1 

#funcion que guarda en fit los visual words y luego predice cada uno de los descriptores
def kMeans(descriptores, k, archivoSalida):        

    #convertir a 1 fila
    descriptoresFila = descriptores[0][1]
    for nombreArchivo, descriptor in descriptores[1:]:
        descriptoresFila = np.vstack((descriptoresFila, descriptor))

    #convertir a float
    descriptoresFloat = descriptoresFila.astype(float)

    kmean = cluster.MiniBatchKMeans(n_clusters=k)
    kmean = kmean.fit(descriptoresFloat)

    joblib.dump((kmean,k),archivoSalida)

    return kmean
        
def crearHistograma(kmean, descriptores, k):
    histogramas = np.zeros((len(descriptores),k),"float32")
    for i in range(len(descriptores)):
        predicciones =  kmean.predict(descriptores[i][1])
        for prediccion in predicciones:
            histogramas[i][prediccion] += 1

    return histogramas

def entrenar(descriptores, clases, codeBookSalida, clasificadorSalida):
    k = 192

    codeBook = kMeans(descriptores,k,codeBookSalida)    
    
    x_entrenamiento = crearHistograma(codeBook,descriptores,k)
    y_entrenamiento = clases

    #con datos del cross validation
    randomForest = ensemble.RandomForestClassifier(n_estimators=69,max_depth=26,criterion="gini",max_features=3)
    randomForest.fit(x_entrenamiento,y_entrenamiento)    
    y_pred = randomForest.predict(x_entrenamiento)

    joblib.dump((randomForest),clasificadorSalida)

    estadisticasPorClase(y_entrenamiento,y_pred)

def main(argv):
    descriptores = joblib.load(argv[0])
    clases = leerJSON(argv[1])
    codeBookSalida = argv[2]
    clasificadorSalida = argv[3]

    entrenar(descriptores,clases, codeBookSalida, clasificadorSalida)

if __name__ == "__main__":
    main(sys.argv[1:])
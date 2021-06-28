import sys
import json
import pandas as pd
import joblib
from sklearn import metrics, svm, preprocessing

#leer las clases de cada foto
def leerJSON(nombreArchivo):
    clases = []
    with open(nombreArchivo, 'r') as file:
        jsonData = file.read()
        etiquetas = json.loads(jsonData)
        for etiqueta in etiquetas:
            clases.append(etiquetas[etiqueta]["perteneceA"])
    return clases

#calcula las estadisticas de cada clase (precision, recall, f1-score)
def estadisticasPorClase(y_prueba, y_pred):

    matrizConfusion = metrics.confusion_matrix(y_prueba,y_pred)
    
    nuevaMatriz = [[0,0],[0,0]]
    clase = "Equipo A"
    for i in range(3):
        pos11 = matrizConfusion[i][i]
        pos00 = matrizConfusion[0][0] + matrizConfusion[1][1] + matrizConfusion[2][2] - pos11                
        pos01 = 0
        pos10 = 0
        for j in range(3):
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
            clase = "Equipo B"
        elif i == 2:
            clase = "Ninguno"

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
    print("Recall Promedio: ",round(metrics.recall_score(y_prueba,y_pred,labels=["Equipo A","Equipo B","Ninguno"],average="macro")*100,5))
    #preciosion promedio
    print("Precision Promedio: ",round(metrics.precision_score(y_prueba,y_pred,labels=["Equipo A","Equipo B","Ninguno"],average="macro")*100,5))
    #f1-socre promedio
    print("F1-Score Promedio: ",round(metrics.f1_score(y_prueba,y_pred,labels=["Equipo A","Equipo B","Ninguno"],average="macro")*100,5))

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
        
def evaluar(clasificador, scaler, x, y):
    
    datosNormalizados = scaler.transform(x)

    x_prueba = datosNormalizados
    y_prueba = y

    y_pred = clasificador.predict(x_prueba)

    estadisticasPorClase(y_prueba,y_pred)

def main(argv):
    colorFotos = pd.read_csv(argv[0])
    clases = leerJSON(argv[1])    
    scaler, clasificador = joblib.load(argv[2])

    evaluar(clasificador,scaler,colorFotos,clases)
    

if __name__ == "__main__":
    main(sys.argv[1:])
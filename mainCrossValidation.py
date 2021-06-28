import sys
import pandas as pd
import json
import numpy as np
from sklearn import metrics, svm, preprocessing, utils
from random import randint

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

    print(matrizConfusion)
    
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

def crossValidation(modelo, x , y, inFolds):
    x_aleatoria, y_aleatoria = utils.shuffle(x, y)
    f1_score_global = 0
    accuracy_global = 0
    cantElem = int(len(x) / inFolds)    

    for i in range(0,inFolds):
        print()
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Iteracion :",(i+1))

        validacion_x = x_aleatoria[(i*cantElem):(i*cantElem + cantElem)]            
        validacion_y = y_aleatoria[(i*cantElem):(i*cantElem + cantElem)]        

        training_x = x_aleatoria[:(i*cantElem)] + x_aleatoria[(i*cantElem + cantElem):]
        training_y = y_aleatoria[:(i*cantElem)] + y_aleatoria[(i*cantElem + cantElem):]  

        scaler = preprocessing.StandardScaler()
        training_x = scaler.fit_transform(training_x)
        validacion_x = scaler.transform(validacion_x)

        modelo.fit(training_x,training_y)
        y_pred = modelo.predict(validacion_x)

        f1_score = metrics.f1_score(validacion_y,y_pred,labels=["Equipo A","Equipo B","Ninguno"],average="macro")        
        accuracy = metrics.accuracy_score(validacion_y,y_pred)        

        estadisticasPorClase(validacion_y,y_pred)
        print("Score:",f1_score)
        print("Accuracy:",accuracy)
        if not pd.isna(f1_score):
            f1_score_global += f1_score
        if not pd.isna(accuracy):
            accuracy_global += accuracy

    f1_score_global /= inFolds
    print("Accuracy Global = ",accuracy_global / inFolds)
    return f1_score
           
#funcion que genera una iteracion de los paramtros unica
def generarIteracionUnica(iteraciones):

    c = round(np.random.uniform(0.1,100.0),1)
    gamma = round(np.random.uniform(0.0001,10.0),4)

    print(c,gamma)    

    nuevaIteracion = [c, gamma]

    while not esIteracionUnica(iteraciones, nuevaIteracion):
        c = round(np.random.uniform(0.1,100.0),1)
        gamma = round(np.random.uniform(0.0001,10.0),4)
        nuevaIteracion = [c, gamma]

    return nuevaIteracion

#funcion que revisa que los parametros usados sean diferentes
def esIteracionUnica(iteraciones, nuevaIteracion):
    for fila in iteraciones:
        esIgual = True
        for index, columna in enumerate(fila):
            if nuevaIteracion[index] != columna:
                esIgual = False
        if esIgual:
            return False
    return True

#ajustar los hiper parametros con n-fold cross-validation
def busquedaParametros(coloresFotos,y_entrenamiento,inFolds):
    iteraciones = []   

    mayorPromedio =  0.9301
    nuevoPromedio = 0
    cont = 1
    
    while nuevoPromedio < mayorPromedio:      
        iteracion = generarIteracionUnica(iteraciones)   
        iteraciones.append(iteracion)

        svc = svm.SVC(C=iteracion[0],gamma=iteracion[1])
        print("---------------------------------------------")
        print("Configuracion ",cont)
        cont+=1
        score = crossValidation(svc,coloresFotos, y_entrenamiento,inFolds)
        print(iteracion)
        print("Promedio:",score)
        nuevoPromedio = score    
        if nuevoPromedio > mayorPromedio:
            mejorIteracion = iteracion
                
    print()
    print("------------------------------------")
    print("Mayor Promedio:",nuevoPromedio)
    print("Mejor Iteracion:",mejorIteracion)

#c,gamma
#20.3, 5.5712
#31.7, 2.7678
#41.2, 5.971

def main(argv):          
    colorFotos = pd.read_csv(argv[0])
    clases = leerJSON(argv[1])
    inFolds = int(argv[2])  
    busquedaParametros(colorFotos.values.tolist(),clases,inFolds)    


if __name__ == "__main__":
    main(sys.argv[1:])
import cv2
import sys
import numpy as np

def deteccionPlayerconBalon(CordenadasJugadores, cordenadaPelota, pelotaD, colors, img, font, pelotaNod, pos):
    # Comparacion de distrancias
    jugadorConBalonEncontrado = []
    playersOnscreen = len(CordenadasJugadores)
    print("cuantos :" + str(len(CordenadasJugadores)))
    if len(cordenadaPelota):
        pelotaD = pelotaD + 1
        for i in range(playersOnscreen):
            color = colors[i]
            x1, y1, w1, h1 = cordenadaPelota
            x2, y2, w2, h2 = CordenadasJugadores[i]

            if x1 < x2:
                distancia_pixeles = abs(x2 - (x1+w1))
                if distancia_pixeles < 20:
                    pos = pos + 1
                    # Dice que jugador tiene la pelota
                    print("Jugador " + str(i) +
                          "tiene la pelota, pixels:" + str(distancia_pixeles))
                    cv2.rectangle(img, (x2-(x2 - x1), y2),
                                  (x2+w2, y2+h2), color, 2)
                    cv2.putText(img, "Player con pelota",
                                (x2, y2+20), font, 2, (255, 255, 255), 2)
                    jugadorConBalonEncontrado = CordenadasJugadores[i]
            else:
                distancia_pixeles = abs(x1 - (x2+w2))
                if distancia_pixeles < 20:
                    pos = pos + 1
                    # Dice que jugador tiene la pelota
                    print("Jugador  " + str(i) +
                          "tiene la pelota, pixels:" + str(distancia_pixeles))
                    cv2.rectangle(
                        img, (x2, y2), (x1+w1, y1+h1), color, 2)
                    cv2.putText(img, "Player con pelota",
                                (x2, y2+20), font, 2, (255, 255, 255), 2)
                    jugadorConBalonEncontrado = CordenadasJugadores[i]
    else:
        pelotaNod = pelotaNod + 1
    return pelotaNod, pelotaD, pos, jugadorConBalonEncontrado

def procesamientoVideo(video):
    net = cv2.dnn.readNet('./Yolo/yolov3.weights', './Yolo/yolov3.cfg')
    classes = []
    frames = 0
    pos = 0
    pelotaD = 0
    pelotaNod = 0
    with open('./Yolo/coco.names', 'r') as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture(video)
    # img = cv2.imread('image2.jpg')

    while True:
        frames = frames + 1
        _, img = cap.read()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
        bboxes = []
        confianzas = []
        indexClasses = []
        for output in layerOutputs:
            for detection in output:
                # se consideran desde el elemento 6 hasta el final porque
                # los primeros 4 son las coordenadas del bounding box
                # y la posicion 5 es la confianzas que tiene el bounding box de que es X objeto
                # el resto de posiciones son las probabilidades por classes, o sea los nombres que tiene el archivo coco.names

                # se extrae la clase que tiene la mayor confianzas que se detecto
                indexClass = np.argmax(detection[5:])
                # se extrae la confianzas/probabilidad que se obtuvo en esa clase
                confianzasEnDetectar = detection[5:][indexClass]
                if confianzasEnDetectar > 0.5:
                    # se multiplica por el width y height respectivamente porque habiamos normalizado la imagen
                    # con blob, entonces queremos obtener el resulado original de la imagen
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # YOLO trabaja con el centro de los bounding boxes, debemos de sacar las coordenadas de la
                    # esquina superior izquierda para poder mostrarlas con openCV
                    x = int(center_x-w/2)
                    y = int(center_y-h/2)
                    # se guardan las coordenadas de cada bbox,confianza del mismo y a la clase que pertence
                    # clase siendo Persona o Balon en este caso
                    bboxes.append([x, y, w, h])
                    confianzas.append(float(confianzasEnDetectar))
                    indexClasses.append(indexClass)

        cordenadaPelota = []
        CordenadasJugadores = []
        # NMSBoxes quita todos esos bounding boxes que estan unos encima de otros, y solo deja 1 para dicho objeto
        indexes = cv2.dnn.NMSBoxes(bboxes, confianzas, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        print("----------------------")
        print(indexes)
        print("------------------")
        print(indexes.flatten())
        colors = np.random.uniform(0, 255, size=(len(bboxes), 3))
        if len(indexes) > 0:
            # se usa el metodo flatten para que la matriz extraida se haga una lista y poder recorrerla
            for j in indexes.flatten():
                x, y, w, h = bboxes[j]
                label = str(classes[indexClasses[j]])                
                        
                # cordenadas para medicion
                if(label == "sports ball"):
                    cordenadaPelota = [x, y, w, h]
                else:
                    CordenadasJugadores.append([x, y, w, h])
                
            for i in indexes.flatten():
                # se extraen las coordenadas de cada bbox
                x, y, w, h = bboxes[i]
                # se obtiene el texto si es persona o un balon
                label = str(classes[indexClasses[i]])
                if(label == "person" or label == "sports ball"):
                    confianzaDetectada = str(round(confianzas[i], 2))
                    color = colors[i]
                    # metodo para la deteccion del player con Balon
                    pelotaNod, pelotaD, pos, jugadorConBalon = deteccionPlayerconBalon(
                        CordenadasJugadores, cordenadaPelota, pelotaD, colors, img, font, pelotaNod, pos)
                    # se colorea el BBox si es un jugador que no tiene el balon
                    #if jugadorConBalon != bboxes[i] and bboxes[i] != cordenadaPelota:
                        #cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        #cv2.putText(img, label+" "+confianzaDetectada, (x, y+20),
                        #            font, 2, (255, 255, 255), 2)                      

        print("Frames:")
        print(frames)
        print("Se vio la pelota en esta cantidad de frames:")
        print(pelotaD)
        print("No se vio la pelota en esta cantidad de frames:")
        print(pelotaNod)
        print("Jugador controlaba la pelota en esta cantidad de frames:")
        print(pos)
        imS = cv2.resize(img, (960, 540))
        cv2.imshow('Image', imS)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main(argv):
    procesamientoVideo(argv[0])

if __name__ == "__main__":
    main(sys.argv[1:])
import os
import json
import sys

def renombrarArchivos(directorio):
    for count, filename in enumerate(sorted(os.listdir(directorio))):
        if count < 10:
            nuevoNombre = "player00"
        elif count >= 10 and count <= 99 :
            nuevoNombre = "player0"
        elif count >= 100 : 
            nuevoNombre = "player"
        nuevoNombre += str(count) + ".jpg"
        original = directorio+filename
        destino = directorio+nuevoNombre
        os.rename(original,destino)

def genArchivoEtiquetas(directorio,archivoSalida):
    objeto = {}
    i = 0
    for filename in sorted(os.listdir(directorio)):
        nombreFoto = filename
        objeto[nombreFoto] = { "perteneceA":""}

    with open(archivoSalida, "w") as file:
        json.dump(objeto, file, indent=3)

def main(argv):
    directorio = argv[0]
    archivoSalida = argv[1]
    #renombrarArchivos(directorio)
    genArchivoEtiquetas(directorio, archivoSalida)

if __name__ == "__main__":
    main(sys.argv[1:])
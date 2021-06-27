import os
import json

def renombrarArchivos():
    dir = "./JugadoresAZUL-ROSA/"
    for count, filename in enumerate(sorted(os.listdir(dir))):
        if count < 10:
            nuevoNombre = "sa0"
        else :
            nuevoNombre = "sa"
        nuevoNombre += str(count) + ".jpg"
        original = dir+filename
        destino = dir+nuevoNombre
        os.rename(original,destino)

def genArchivoEtiquetas():
    objeto = {}
    i = 0
    for filename in sorted(os.listdir("./JugadoresAZUL-ROSA/")):
        nombreFoto = filename
        objeto[nombreFoto] = { "perteneceA":""}

    with open("training_etiquetas.json", "w") as file:
        json.dump(objeto, file, indent=3)

def main():
    #renombrarArchivos()
    genArchivoEtiquetas()

if __name__ == "__main__":
    main()
import math
import numpy as np
import matplotlib.pyplot as plt

def test(Xtest,ydTest,W,capas,part):

    cantPruebas = len(Xtest)
    cantCapas = len(capas)
    e = math.e
    arraySalidas = []
    columnBIAS = (np.full((cantPruebas, 1), -1))
    Xtest = np.hstack((columnBIAS, Xtest))  # reagrupamos
    b=1
    error = 0
    for i in range(cantPruebas):
        yCapaBIAS = []
        yCapaBIAS.insert(0, Xtest[i][:].tolist())  # Insertamos las entradas
        for j in range(cantCapas):
            yAux = []
            Xt = np.transpose(yCapaBIAS[j][:])
            V = np.dot(W[j][:][:], Xt)
            for k in range(len(V)):
                sigmoide = (2 / (1 + e ** (-b * V[k]))) - 1
                yAux.insert(k, sigmoide)
            yAux.insert(0, -1)
            yCapaBIAS.insert(j + 1, yAux)
        y = yCapaBIAS[cantCapas][:]
        # Regla de y
        y = np.array(y[1:])
        index = np.unravel_index(y.argmax(), y.shape)
        for l in range(len(y)):
            y[l] = -1
        y[index]=1
        arraySalidas.insert(i,y)
        # Calculo del error promedio en una epoca
        cont=0
        #print("Comparo nuestro y", y, " - ", yd[i])
        for m in range(len(y)):
            if y[m] != ydTest[i][m]:
                cont+=1
                #print("Hubo un error")
        if cont!=0:
            error = error + 1

    error = error / cantPruebas
    plt.figure(part+151)
    plt.title('Componentes principales para IRIS')
    plt.xlabel("Promedio ancho")
    plt.ylabel("Promedio largo")
    vColor = ["green", "red", "blue"]
    tipoPlanta = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

    for i in range(cantPruebas):
        promLargo = (Xtest[i][1] + Xtest[i][2]) / 2
        promAncho = (Xtest[i][3] + Xtest[i][4]) / 2
        if (np.equal(arraySalidas[i], tipoPlanta[0])).all():
            color = 0
        elif (np.equal(arraySalidas[i], tipoPlanta[1])).all():
            color = 1
        elif (np.equal(arraySalidas[i], tipoPlanta[2])).all():
            color = 2
        plt.plot(promAncho, promLargo, marker='+', color=vColor[color])
    plt.savefig(f'Valores{part}.jpg')


    return error
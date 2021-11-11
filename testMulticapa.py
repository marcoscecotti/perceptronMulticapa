import math
import numpy as np
from matplotlib import pyplot as plt


def test(df,W,capas):
    X = df.iloc[:, 0:-1]
    yd = df.iloc[:, -1]

    X = X.to_numpy()
    yd = yd.to_numpy()

    cantPruebas = len(X)
    cantEntradas = len(X[0])
    cantCapas = len(capas)
    e = math.e

    columnBIAS = (np.full((cantPruebas, 1), -1))
    X = np.hstack((columnBIAS, X))  # reagrupamos
    b=1
    error = 0
    vectorConfusion = []
    for i in range(cantPruebas):
        yCapaBIAS = []
        yCapaBIAS.insert(0, X[i][:].tolist())  # Insertamos las entradas
        for j in range(cantCapas):
            yAux = []
            Xt = np.transpose(yCapaBIAS[j][:])
            V = np.dot(W[j][:][:], Xt)
            for k in range(len(V)):
                sigmoide = (2 / (1 + e ** (-b * V[k]))) - 1
                yAux.insert(k, sigmoide)
            yAux.insert(0, -1)
            yCapaBIAS.insert(j + 1, yAux)
        y = yCapaBIAS[cantCapas][1]  # Salida final
        # Funcion signo
        if y > 0:
            y = 1
        else:
            y = -1

        if y != yd[i]:
            error = error + 1

        # Calculo de la matriz de confusion
        if y == 1 and yd[i] == 1:
            vectorConfusion.insert(i, 0)
        if y == 1 and yd[i] == -1:
            vectorConfusion.insert(i, 1)
        if y == -1 and yd[i] == -1:
            vectorConfusion.insert(i, 2)
        if y == -1 and yd[i] == 1:
            vectorConfusion.insert(i, 3)

    error = error / cantPruebas

    plt.title('Datos en concentile TEST.csv')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    vColor = ["green", "red", "blue", "black"]
    for i in range(cantPruebas):
        plt.plot(X[i][1], X[i][2], marker='+', color=vColor[vectorConfusion[i]])
    plt.show()
    return error
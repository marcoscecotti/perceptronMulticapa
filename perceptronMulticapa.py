import math
import numpy as np

def perceptronMulticapa(X, yd, capas, momento):
    tasaAcierto = []
    cantCapas = len(capas)

    cantPruebas = len(X)
    cantEntradas = len(X[0])

    # Le agrego la entrada -1 correspondiente al BIAS (umbral / parcialidad)
    columnBIAS = (np.full((cantPruebas, 1), -1))
    X = np.hstack((columnBIAS, X))  # reagrupamos

    W = []
    # Inicializamos un vector de matrices con valores entre -0.5 y 0.5
    i = 0
    W.insert(i, (np.random.rand((capas[i]), (cantEntradas + 1))) - 0.5)

    for i in range(1, cantCapas):
        W.insert(i, (np.random.rand((capas[i]), (capas[i - 1] + 1))) - 0.5)

    b = 1
    e = math.e
    epocas = 0
    maxEpocas = 1000
    tol = 90
    acierto = 0
    while (epocas < maxEpocas and acierto < tol):

        # Para cada prueba
        for i in range(cantPruebas):
            yCapaBIAS = []
            yCapaBIAS.insert(0, X[i][:])  # Vector con las salidas de cada capa, incluido el BIAS en la posicion 0

            # Propagacion hacia adelante
            for j in range(cantCapas):
                yAux = []
                Xt = np.transpose(yCapaBIAS[j][:])
                V = np.dot(W[j][:][:], Xt)
                for k in range(len(V)):
                    yAux.insert(k, (2 / (1 + e ** (-b * V[k]))) - 1)
                yAux.insert(0, -1)
                yCapaBIAS.insert(j + 1, yAux)

            # Calculamos el vector de deltas
            # Quitamos el -1 de las salidas
            yCapa = []
            for k in range(len(yCapaBIAS)):
                yCapa.insert(k, yCapaBIAS[k][1:len(yCapaBIAS[k])])

            # Inicializo el vector de deltas del tamaño de la cantidad de neuronas que hay en cada capa
            vDelta = []
            for z in range(0, cantCapas):
                vDelta.insert(z, (np.zeros(capas[z])))

            # Retropropagacion
            for j in range(cantCapas - 1, -1, -1):
                if (j == cantCapas - 1):  # En la ultima capa se calcula yd-yAproximado
                    delta = []
                    for k in range(capas[cantCapas - 1]):
                        error = yd[i][k] - yCapa[cantCapas][k]
                        delta.insert(k, error * (1 / 2) * (1 + yCapa[cantCapas][k]) * (1 - yCapa[cantCapas][k]))
                    vDelta[j] = delta
                else:
                    wTSinBIAS = np.empty((len(W[j + 1][:][:]), capas[j]))
                    # wTSinBIAS contiene la matriz de la capa j, sin BIAS y transpuesta
                    for k in range(len(W[j + 1][:][:])):
                        wTSinBIAS[k][:] = W[j + 1][k][1:capas[j] + 1]
                    wTSinBIAS = np.transpose(wTSinBIAS)
                    delta = []
                    WDelta = np.dot(wTSinBIAS[:][:],vDelta[j + 1][:])  # Multiplicacion matriz-vector -> Sumatoria W*Delta
                    for k in range(capas[j]):
                        deltaActual = WDelta[k] * (1 / 2) * (1 + yCapa[j + 1][k]) * (1 - yCapa[j + 1][k])
                        delta.insert(k, deltaActual)
                    vDelta[j] = delta  # En la capa j guardamos los deltas de las k neuronas de esa capa

            # Ajuste de pesos
            n = 0.1  # Tasa de aprendizaje
            for j in range(cantCapas):
                for k in range(capas[j]):  # Para cada neurona
                    deltaW=n*np.dot(vDelta[j][k],yCapaBIAS[j][:])
                    W[j][k][:] = W[j][k][:] + deltaW

        # Comparo las salidas y calculo la cantidad de aciertos en una epoca
        acierto = 0
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

            y = yCapaBIAS[cantCapas][:]  # Salida final
            # Regla de y
            y = np.array(y[1:])
            index = np.unravel_index(y.argmax(), y.shape)
            for l in range(len(y)):
                y[l] = -1
            y[index] = 1

            cont = 0
            for m in range(len(y)):
                if y[m] != yd[i][m]:
                    cont += 1

            if cont == 0:
                acierto+=1
        acierto = (acierto / cantPruebas)*100
        tasaAcierto.insert(epocas,acierto)
        epocas = epocas + 1

    return W,tasaAcierto

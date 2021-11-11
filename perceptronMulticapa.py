import math
import matplotlib.pyplot as plt
import numpy as np

def perceptronMulticapa(df,capas,momento):

    errorEpoca = []

    X = df.iloc[:, 0:-1]
    yd = df.iloc[:, -1]

    X = X.to_numpy()
    yd = yd.to_numpy()

    cantPruebas = len(X)
    cantEntradas = len(X[0])
    cantCapas = len(capas)

    # Le agrego la entrada -1 correspondiente al BIAS (umbral / parcialidad)
    columnBIAS = (np.full((cantPruebas, 1), -1))
    X = np.hstack((columnBIAS, X))  # reagrupamos

    W=[]
    #Inicializamos un vector de matrices con valores entre -0.5 y 0.5
    i=0
    # Mismos valores
    #np.random.seed(666)
    W.insert(i,(np.random.rand((capas[i]), (cantEntradas+1)))-0.5)

    for i in range(1,cantCapas):
        W.insert(i,(np.random.rand((capas[i]), (capas[i-1]+1)))-0.5)

    # Creacion de historialCapa W
    historialDeltaW = []
    i=0
    historialDeltaW.insert(i, (np.zeros((capas[i], cantEntradas + 1))))
    for i in range(1, cantCapas):
        dataInsert = (np.zeros((capas[i], (capas[i - 1] + 1))))
        historialDeltaW.insert(i, dataInsert)

    b=1
    e = math.e
    epocas = 0
    maxEpocas = 700
    umbralError = 0.09
    promError = 1
    while (epocas < maxEpocas and promError > umbralError):

        # Para cada prueba
        for i in range(cantPruebas):
            yCapaBIAS = []
            yCapaBIAS.insert(0,X[i][:]) #Vector con las salidas de cada capa, incluido el BIAS en la posicion 0

            # Propagacion hacia adelante
            for j in range(cantCapas):
                yAux = []
                Xt = np.transpose(yCapaBIAS[j][:])
                V = np.dot(W[j][:][:], Xt)
                for k in range(len(V)):
                    #v=sum(V[k][:])
                    yAux.insert(k, (2/( 1 + e** (-b*V[k]) ) ) -1 )
                yAux.insert(0, -1)
                yCapaBIAS.insert(j+1, yAux)

            #Calculamos el vector de deltas
            #Quitamos el -1 de las salidas
            yCapa=[]
            for k in range(len(yCapaBIAS)):
                yCapa.insert(k,yCapaBIAS[k][1:len(yCapaBIAS[k])])

            #Inicializo el vector de deltas del tamaño de la cantidad de neuronas que hay en cada capa
            vDelta=[]
            for z in range(0, cantCapas):
                vDelta.insert(z, (np.zeros(capas[z])))

            #Retropropagacion
            for j in range(cantCapas-1,-1,-1):
                if(j==cantCapas-1): #En la ultima capa se calcula yd-yAproximado
                    delta = []
                    error = yd[i]-yCapa[cantCapas][0]
                    delta.insert(0,error*(1/2)*(1+yCapa[cantCapas][0])*(1-yCapa[cantCapas][0]))
                    vDelta[j]=delta
                else:
                    sss=len(W[j + 1][:][:])
                    wTSinBIAS = np.empty((len(W[j+1][:][:]),capas[j])) #wTSinBIAS contiene la matriz de la capa j, sin BIAS y transpuesta
                    for k in range(len(W[j+1][:][:])):
                        wTSinBIAS[k][:] = W[j+1][k][1:capas[j]+1]
                    wTSinBIAS = np.transpose(wTSinBIAS)
                    delta = []
                    WDelta=np.dot(wTSinBIAS[:][:], vDelta[j + 1][:]) #Multiplicacion matriz-vector -> Sumatoria W*Delta
                    for k in range(capas[j]):
                        deltaActual = WDelta[k]*(1/2)*(1+yCapa[j+1][k])*(1-yCapa[j+1][k])
                        delta.insert(k,deltaActual)
                    vDelta[j]=delta #En la capa j guardamos los deltas de las k neuronas de esa capa


            #Ajuste de pesos
            n = 0.1  # Tasa de aprendizaje
            alpha = 0.4
            for j in range(cantCapas):
                for k in range(capas[j]): #Para cada neurona
                    if momento == 0:
                        deltaW = n * np.dot(vDelta[j][k], yCapaBIAS[j][:])
                    else:
                        deltaW = n*np.dot(vDelta[j][k],yCapaBIAS[j][:])+alpha*historialDeltaW[j][k]
                        historialDeltaW[j][k] = deltaW
                    W[j][k][:]=W[j][k][:]+deltaW

        # Comparo las salidas y calculo la cantidad de errores en una epoca
        error = 0
        vectorConfusion = []
        for i in range(cantPruebas):
            yCapaBIAS = []
            yCapaBIAS.insert(0, X[i][:].tolist()) # Insertamos las entradas
            for j in range(cantCapas):
                yAux = []
                Xt = np.transpose(yCapaBIAS[j][:])
                V = np.dot(W[j][:][:], Xt)
                for k in range(len(V)):
                    # v = sum(V[k][:])
                    sigmoide = (2 / (1 + e ** (-b * V[k]))) - 1
                    yAux.insert(k, sigmoide)
                yAux.insert(0, -1)
                yCapaBIAS.insert(j + 1, yAux)
            y = yCapaBIAS[cantCapas][1] #Salida final
            #Funcion signo
            if y > 0:
                y = 1
            else:
                y = -1

            # Calculo de la matriz de confusion
            if y==1 and yd[i]==1:
                vectorConfusion.insert(i,0)
            if y==1 and yd[i]==-1:
                vectorConfusion.insert(i,1)
            if y==-1 and yd[i]==-1:
                vectorConfusion.insert(i,2)
            if y==-1 and yd[i]==1:
                vectorConfusion.insert(i,3)

            # Calculo del error promedio en una epoca
            if y!=yd[i]:
                error=error+1

        promError = error / cantPruebas
        errorEpoca.insert(epocas,promError)
        epocas = epocas + 1

    plt.title('Error vs Epoca')
    plt.xlabel("Epocas")
    plt.ylabel("Error")
    plt.plot(errorEpoca, color='g')
    plt.show()

    plt.title('Datos en concentile TRAIN.csv')
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
    return W,promError,epocas
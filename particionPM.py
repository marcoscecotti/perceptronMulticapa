from testPM import perceptronMulticapaTest

def particion(indices, df, capas):
    arrayErrores = []
    arrayErrorEpocas = []
    for i in range(len(indices)):  # Cantidad de particiones
        train = df.iloc[indices[i][0], :]
        test = df.iloc[indices[i][1], :]
        error,errorEpoca = perceptronMulticapaTest(train, test, capas,i)  # Cantidad de errores (Perceptron + test)
        arrayErrores.insert(i, error)
        arrayErrorEpocas.insert(i,errorEpoca)
    return arrayErrores, arrayErrorEpocas

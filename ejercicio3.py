import math
from sklearn.model_selection import train_test_split
from perceptronMulticapa import perceptronMulticapa
from testMulticapa import test
import pandas as pd
from perceptron import perceptronSimple

#Leo el archivo CSV
df = pd.read_csv('icgtp1datos/XOR_trn.csv', header=None)

#Lo desordeno y extraigo el 80% de los datos para train y el 20% para test
train1, test1 = train_test_split(df, test_size = 0.20,random_state=42)

capas = [2,1]
#Inciso 1
W,promErrorTrain,epocas = perceptronMulticapa(train1,capas,0)
promErrorTest = test(test1,W,capas)
print('Perceptron multicapa sin momento')
print('Error de entrenamiento: ',promErrorTrain)
print('Error de test: ',promErrorTest)
print('Epocas sin momento: ',epocas)

#Inciso 2
W,promErrorTrain,epocas = perceptronMulticapa(train1,capas,1)
promErrorTest = test(test1,W,capas)
print('Perceptron multicapa con momento')
print('Error de entrenamiento: ',promErrorTrain)
print('Error de test: ',promErrorTest)
print('Epocas con momento: ',epocas)

# Inciso 3
# X = df.iloc[:, 0:-1]
# yd = df.iloc[:, -1]
# X = X.to_numpy()
# yd = yd.to_numpy()
# cantPruebas = len(X)
# cantEntradas = len(X[0])
# cantCapas = len(capas)
#
# # Suma
# promX1 = 0
# promX2 = 0
# for i in range(cantPruebas):
#     promX1 += X[i][0]
#     promX2 += X[i][1]
#
# centro = [promX1/cantPruebas, promX2/cantPruebas]
# entradas = []
# distancias = []
# for i in range(cantPruebas):
#     punto = [X[i][0], X[i][1]]
#     entradas.insert(i, [math.dist(centro, punto), yd[i]])
#
# df = pd.DataFrame(entradas)
# print('Perceptron simple')
# historialW, __ = perceptronSimple(df)

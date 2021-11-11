from validacionCruzada import validacionCruzada
from particionPM import particion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('icgtp1datos/irisbin.csv', header=None)
capas = [2,3]


size = len(df) # Cantidad de pruebas
k = 1 # Particiones de entrenamiento

indices = validacionCruzada(size, k)
errores, erroresEpocas = particion(indices, df, capas)

media = np.mean(errores)
desvio = np.std(errores)

print("Media:", media, " Desvio:", desvio)
print(errores) # Promedio de errores por particion

particiones = int(size/k)
for i in range(particiones):
    plt.figure(i)
    plt.title('Error vs Epoca')
    plt.xlabel("Epocas")
    plt.ylabel("Error")
    plt.plot(erroresEpocas[i], color='g')
    plt.savefig(f'ErrorEpocas{i}.jpg')


# 10 particiones:
# Media: 0.08
# Desvio: 0.075
#
# 1 particion:
# Media: 0.04
# Desvio: 0.19
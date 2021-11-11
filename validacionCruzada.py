import pandas as pd
import numpy as np

# n: Cantidad de pruebas
# k: Cantidad de datos de test

def validacionCruzada(n, k):
    random_data = np.random.choice(n, n, replace=False)
    particiones = int(n / k)
    indices = []
    data_trnOut = []
    data_tstOut = []
    random_dataAux = []
    for i in range(particiones):
        data_tstOut = random_data[0:k]
        data_trnOut = random_data[k:]
        random_dataAux = np.concatenate((random_data[k:], random_data[0:k]), axis=0)
        random_data = []
        random_data = random_dataAux
        dataset = [data_trnOut, data_tstOut]
        indices.insert(i, dataset)

    return indices



import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import funciones


"""## Recortamos en ventanas de 8s

8s serían 16000 muestras

Normal → stride: 3s

Anormal → stride: 8s
"""

def CortarSeñal(ecg, longitud, stride_normal, stride_anormal, fs, normal, anormal):
  ventanas_ecg_normal = []
  ventanas_ecg_anormal = []

  ventanas_pcg_normal = []
  ventanas_pcg_anormal = []
  for i in range(len(ecg)):
    try:
      #fs = ecg[i]['fs']
      #longitud = fs*8
      if(i in anormal):
        #stride = fs*8
        ventanas_pcg_anormal.append(funciones.Recortar(ecg[i]['Filtrado PCG'], longitud, stride_anormal, fs))
        ventanas_ecg_anormal.append(funciones.Recortar(ecg[i]['Filtrado ECG'], longitud, stride_anormal, fs))
      else:
        #stride = fs*3
        ventanas_pcg_normal.append(funciones.Recortar(ecg[i]['Filtrado PCG'], longitud, stride_normal, fs))
        ventanas_ecg_normal.append(funciones.Recortar(ecg[i]['Filtrado ECG'], longitud, stride_normal, fs))
    except IndexError as e:
      print(f"Error en la iteración {i}: {e}")


  data = []
  for i in range(len(ventanas_pcg_normal)):
    for j in range(len(ventanas_pcg_normal[i])):
      data.append({
          'pcg': ventanas_pcg_normal[i][j],
          'ecg': ventanas_ecg_normal[i][j],
          'condicion': 'Normal',
          'fs': 2000
      })

  for i in range(len(ventanas_pcg_anormal)):
    for j in range(len(ventanas_pcg_anormal[i])):
      data.append({
          'pcg': ventanas_pcg_anormal[i][j],
          'ecg': ventanas_ecg_anormal[i][j],
          'condicion': 'Anormal',
          'fs': 2000
      })
  return data


"""## Segmentar por R-R

Este siento que es mejor porque en el Guyton en el dibujo del ciclo cardíaco pareciera que el final de la onda T coincide con el final de S2, pero en otros papers dice que el final de la onda T coincide con el inicio de S2

El problema de cortarlo así es que no resuelve el tema del desbalance de clases porque cortar por latidos por cantidad de normal o anormal. Lo que cambia el desbalance es el stride en el otro caso
"""

def CortarLatidos(ecg, peaks_R, normal, anormal, fs):
  
  latidos_normal = []
  latidos_anormal = []
  for i in range(len(ecg)):
    try:
      if(i in anormal):
        latidos_anormal.append(funciones.RecortarRR(ecg[i]['Filtrado PCG'], ecg[i]['Filtrado ECG'], peaks_R[i], fs)[0])
      else:
        latidos_normal.append(funciones.RecortarRR(ecg[i]['Filtrado PCG'], ecg[i]['Filtrado PCG'], peaks_R[i], fs)[0])
    except IndexError as e:
      print(f"Error en la iteración {i}: {e}")

  latidos = []
  for i in range(len(latidos_normal)):
    for j in range(len(latidos_normal[i])):
      latidos.append({
          'pcg': latidos_normal[i][0][j],
          'ecg': latidos_normal[i][1][j],
          'condicion': 'Normal',
          'fs': 2000
      })

  for i in range(len(latidos_anormal)):
    for j in range(len(latidos_anormal[i])):
      latidos.append({
          'pcg': latidos_normal[i][0][j],
          'ecg': latidos_normal[i][1][j],
          'condicion': 'Anormal',
          'fs': 2000
      })
  return latidos

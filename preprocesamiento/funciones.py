import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfiltfilt, butter, find_peaks
import pandas as pd
import seaborn as sns
from math import log
import neurokit2 as nk
import pywt
import scipy as sp
import os
from scipy.io import wavfile
from scipy.signal import spectrogram


"""# Funciones"""
#################################################################################
############################# Importar señales ##################################
#################################################################################

def importarECG(i):
  dat_file = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Dataset/archive/training-a/a0'+str(i)+'.dat'
  hea_file = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Dataset/archive/training-a/a0'+str(i)+'.hea'
  if os.path.exists(dat_file) and os.path.exists(hea_file):
    fs = 2000
    num_signals = 1
    # Read the header information from the .hea file
    with open(hea_file, 'r') as header_file:
        for line in header_file:
            if line.startswith('#'):
                condicion = line.split()[-1]
            if line.startswith('Sampling frequency'):
                fs = float(line.split()[-1])
            if line.startswith('# 0'):
                break

    # Read the binary signal data from the .dat file
    with open(dat_file, 'rb') as data_file:
        signal_data = np.fromfile(data_file, dtype=np.int16)

    # Create a time array based on the sampling frequency
    time = np.arange(len(signal_data)) / fs

    # Reshape the data to match the number of signals
    signal_data = signal_data.reshape(-1, 1)

    return signal_data[:,0], time, fs, condicion
  

def importarPCG(wav_file, hea_file):

  if os.path.exists(wav_file):
    print('dentro de la funcion')
    # Read the .wav file
    sampling_rate, audio_data = wavfile.read(wav_file)

    # Create a time array based on the sampling rate
    time = [j / sampling_rate for j in range(len(audio_data))]

  return audio_data, time, sampling_rate

#################################################################################
################################## Generales ####################################
#################################################################################
def Maximo(signal):
  max=0
  maxpos=0
  for i in range(len(signal)):
    if(abs(signal[i])>max):
      max = abs(signal[i])
      maxpos=i
  return max, maxpos

def tiempo(signal, fs):
  t = np.arange(len(signal)) / fs
  return t

def Promedio(lista):
  sum = 0
  for i in range(len(lista)):
    sum += lista[i]
  if len(lista)!=0: prom = sum/len(lista)
  else: prom = None
  return prom

def Parametros(key, ecg):
  for i in range(len(ecg)):
    if(i != 185 and i!= 207):
      print(i)
      ecg[i][key[1]] = list(filter(lambda x: not np.isnan(x), ecg[i][key[1]]))
      ecg[i][key[0]+'máximo'] = Maximo(ecg[i][key[1]])[0]
      ecg[i][key[0]+'promedio'] = Promedio(ecg[i][key[1]])
      ecg[i][key[0]+'std'] = np.std(ecg[i][key[1]])
      ecg[i][key[0]+'mínimo'] = min(ecg[i][key[1]])
    else:
      ecg[i][key[0]+'máximo'] = None
      ecg[i][key[0]+'promedio'] = None
      ecg[i][key[0]+'std'] = None
      ecg[i][key[0]+'mínimo'] = None

def InvertirSeñal(ecg, a):
  inverted = []
  for i in range(len(ecg[a]['ecg'])):
    inverted.append(-ecg[a]['ecg'][i])
  return inverted
#################################################################################
################################### Filtros #####################################
#################################################################################
      
"""Pasa bajos"""

def butter_lowpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = sp.signal.filtfilt(b, a, data)
  return y

"""Pasa altos"""

def butter_highpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
  return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
  b, a = butter_highpass(cutoff, fs, order=order)
  y = sp.signal.filtfilt(b, a, data)
  return y

"""Para normalizar"""

def normalize_signal(signal):
  max_val = max(signal)
  min_val = min(signal)
  range_val = max_val - min_val
  normalized_signal = [(2 * (x - min_val) / range_val) - 1 for x in signal]
  return normalized_signal

"""Sacar picos"""
def schmidt_spike_removal(original_signal, fs):
    # Find the window size (500 ms)
    windowsize = int(fs / 2)

    # Find any samples outside of an integer number of windows:
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows:
    sampleframes = np.reshape(original_signal[:-trailingsamples], (windowsize, -1), order='F')

    # Find the MAAs:
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # Create a copy of the original signal to store the despiked signal
    despiked_signal = original_signal.copy()

    # Threshold for spike detection
    threshold = 3 * np.median(MAAs)

    # Iterate through windows to detect and remove spikes
    for i in range(sampleframes.shape[1]):
        if MAAs[i] > threshold:
            # Find the position of the spike within that window:
            spike_position = np.argmax(np.abs(sampleframes[:, i]))

            # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
            zero_crossings = np.concatenate(([False], np.abs(np.diff(np.sign(sampleframes[:, i]))) > 0))

            zero_crossings_before_spike = np.where(zero_crossings[:spike_position])[0]
            if zero_crossings_before_spike.size > 0:
                spike_start = max([0, zero_crossings_before_spike[-1]])
            else:
                spike_start = 0


            zero_crossings_after_spike = np.where(zero_crossings[spike_position:])[0]
            if zero_crossings_after_spike.size > 0:
                spike_end = min(zero_crossings_after_spike[0] + spike_position, windowsize)
            else:
                spike_end = windowsize
            # Set to zero in the copy of the signal
            despiked_signal[i * windowsize + spike_start:i * windowsize + spike_end] = 0.0001

    return despiked_signal

def EncontrarPicosR(data):
  peaks_R = []
  error = []
  for i in range(len(data)):
    try:
      ecg_signal = data[i]['Filtrado ECG']
      _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=2000)
      peaks_R.append(rpeaks['ECG_R_Peaks'])
    except Exception as e:
      error.append(i)
      print(f"An error occurred for sample {i}: {e}")
  return peaks_R, error
#################################################################################
############################### Para graficar ###################################
#################################################################################

def graficar(y,x,titulo, xlabel, ylabel):
  plt.figure(figsize=(25, 8))
  plt.plot(x, y)
  plt.title(titulo)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(True)
  plt.show()

def graficar2(data1, data2, x, label1, label2, titulo, xlabel = 'Tiempo[s]', ylabel = 'Amplitud', color1 = 'crimson', color2 = 'darksalmon'):
  plt.figure()
  plt.plot(x, data1, color=color1, label = label1)
  plt.plot(x, data2,color=color2, label = label2)
  plt.title(titulo)
  plt.legend()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(True)
  plt.show()

def EncontrarPicos(xppg, t_ppg, T, d):
  x=np.zeros(len(xppg))
  for i in range(len(xppg)):
    x[i]=xppg[i]
  picos=find_peaks(x, height = T, distance=d)[0] #le pongo otro parámetro para que no detecte picos falsos. Le pongo una distance para que tenga sentido que no encuentre picos super juntos porque no coincide cpn la frecuencia cardíaca.

  return picos

def GrafPicos(xppg, t_ppg, picos):
  #ahora si elimine los falsos picos.
  plt.figure(figsize=(30,5))
  plt.plot(t_ppg,xppg)
  plt.plot(t_ppg[picos],xppg[picos], 'ro')
  plt.title('ECG con picos')
  plt.ylabel('Amplitud')
  plt.xlabel('Tiempo [s]')
  #plt.xlim([xmin,xmax])
  #plt.ylim([ymin, ymax])
  plt.show()


def GrafLatidoPromedio(ecg, n, window_before = 550, window_after = 1200):
  # Obtener las ubicaciones de los picos R
  r = ecg[n]['picos']['ECG_R_Peaks']
  ecg_signal = ecg[n]['ecg']
  # Definir una ventana alrededor de cada pico R
  #window_before --> Número de puntos antes del pico R
  #window_after --> Número de puntos después del pico R
  #En total se toma como que el latido dura 0.875s, del inicio a R dura  0.275s y de R el final de T dura 0.6s --> Esto habría que buscarlo mejor y fundamentarlo.

  # Extraer los latidos individualmente y rellenar o truncar para tener la misma longitud
  beats = [ecg_signal[peak - window_before : peak + window_after] for peak in r]
  max_length = max(len(beat) for beat in beats)
  beats = [np.pad(beat, (0, max_length - len(beat))) for beat in beats]

  # Convertir la lista de latidos en un array de numpy
  beats_array = np.array(beats)

  # Calcular el promedio de los latidos
  average_beat = np.mean(beats_array, axis=0)

  # Graficar los latidos individualmente
  for beat in beats:
      plt.plot(beat, color='gray', alpha=0.5)

  # Graficar el promedio de los latidos
  plt.plot(average_beat, color='red', label='Promedio de latidos')
  plt.legend()
  plt.title('Latidos individuales y promedio')
  plt.show()

def GrafPeaks(ecg, n, final):
    # Visualize R-peaks in ECG signal
    plot = nk.events_plot(ecg[n]['picos']['ECG_R_Peaks'], ecg[n]['ecg'])

    # Zooming into the first 5 R-peaks
    plot = nk.events_plot(ecg[n]['picos']['ECG_R_Peaks'][:5], ecg[n]['ecg'][:final])
    plt.show()

#################################################################################
############################## Segmentar señales ################################
#################################################################################

def Recortar(data, longitud, stride, fs):
  lista = []
  for i in range(0, len(data), stride):
    if((i+longitud)<=len(data)): lista.append(data[i:i+longitud])
  return lista

def RecortarRR(pcg, ecg, picos_R, fs):
  latidos_pcg = []
  latidos_ecg = []
  for i in range(len(picos_R)-1):
    inicio = int(picos_R[i])
    final = int(picos_R[i+1])
    latidos_pcg.append(pcg[inicio:final])
    latidos_ecg.append(ecg[inicio:final])
  return latidos_pcg, latidos_ecg

#################################################################################
################################## Análisis #####################################
#################################################################################

def Hist(data, x, label, titulo, range, titulo_general):
  # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
  sns.set(style="darkgrid")

  fig, axs = plt.subplots(1, 2, figsize=(15, 5))

  sns.histplot(data=data[0], x=x[0], color="red", label=label[0], ax=axs[0])
  axs[0].set_title(titulo[0])

  sns.histplot(data=data[1], x=x[1], color="skyblue", label=label[1], ax=axs[1])
  axs[1].set_title(titulo[1])

  # Aplicar leyenda y ajustar límites del eje y para ambos subgráficos
  for ax in axs:
      ax.legend()
      ax.set_ylim(range)  # Ajustar los límites del eje y según tus necesidades)
  plt.legend()
  # Añadir un título general encima de los subgráficos
  plt.suptitle(titulo_general, fontsize=16, y=1.02)
  plt.show()

def NorAnor(ecg, key):
  lista = [[], []]
  for i in range(len(ecg)):
    if(ecg[i]['condicion'] == 'Normal'):
      lista[0].append(ecg[i][key])
    else:
      lista[1].append(ecg[i][key])
  return lista

def GrafParametros(key, value, label, titulo, range, titulo_general):
  normal = pd.DataFrame({key[0]: value[0]})
  anormal = pd.DataFrame({key[1]: value[1]})
  print(type(normal), type(anormal))
  Hist(data = [normal, anormal], x = [key[0], key[1]], label = label, titulo = titulo, range = range, titulo_general = titulo_general)

#################################################################################
############################# Phase Space Matrix ################################
#################################################################################

def construct_phase_space_matrix(data, t):
    """
    Construye la matriz de espacio de fase a partir de una secuencia de datos.

    Parámetros:
    - data: Lista o array de datos temporales.
    - t: Número de pasos para calcular la coordenada del espacio de fase (s(n + t) - s(n)).

    Retorna:
    - Matriz de espacio de fase.
    """
    num_points = len(data)
    phase_space_matrix = np.zeros((num_points - t, 2))

    for i in range(num_points - t):
        phase_space_matrix[i, 0] = data[i]
        phase_space_matrix[i, 1] = data[i + t] - data[i]

    return phase_space_matrix


def GrafMatriz(ecg, n, a):
  # Crear subgráficos
    plt.figure(figsize=(12, 6))
    # Primer subgráfico
    plt.subplot(1, 2, 1)
    plt.plot(ecg[n]['Matriz Espacio Fase'][:, 0], ecg[n]['Matriz Espacio Fase'][:, 1], color='b', linestyle='-')
    plt.title('Matriz de Espacio de Fase para Señal de ECG - '+ecg[n]['condicion'])
    plt.xlabel('s(n)')
    plt.ylabel('s(n + t) - s(n)')
    plt.grid(True)

    # Segundo subgráfico
    plt.subplot(1, 2, 2)
    plt.plot(ecg[a]['Matriz Espacio Fase'][:, 0], ecg[a]['Matriz Espacio Fase'][:, 1], color='r', linestyle='-')
    plt.title('Matriz de Espacio de Fase para Señal de ECG - '+ecg[a]['condicion'])
    plt.xlabel('s(n)')
    plt.ylabel('s(n + t) - s(n)')
    plt.grid(True)

    plt.tight_layout()  # Ajusta el espacio entre subgráficos
    plt.show()

#################################################################################
################################### Wavelet #####################################
#################################################################################
# wavelet entropy python
def WE(y, level = 4, wavelet = 'coif2'):
    
    n = len(y)

    sig = y

    ap = {}

    for lev in range(0,level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy

    Enr = np.zeros(level)
    for lev in range(0,level):
        Enr[lev] = np.sum(np.power(ap[lev],2))/n

    Et = np.sum(Enr)

    Pi = np.zeros(level)
    for lev in range(0,level):
        Pi[lev] = Enr[lev]/Et

    we = - np.sum(np.dot(Pi,np.log(Pi)))


    return we
def GrafWavelet(x, freqs, cwtmatr):

    fig, (ax1, ax4) = plt.subplots(2,1, sharex = True, figsize = (30,10))

    t_e = np.arange(0, len(x)) / 2000
    #Señal de ruido
    ax1.plot(t_e, x)
    #ax4.set_ylim(0,1200)
    ax4.set_ylabel("")
    ax4.set_xlabel("Time in [s]")
    ax4.set_title("PCG de un latido")


    # Wavelet transform, i.e. scaleogram
    #scales = np.arange(1, len(x))
    #cwtmatr, freqs = pywt.cwt(x, scales, "morl", sampling_period = 1 / 2000)
    im2 = ax4.pcolormesh(t_e, freqs, cwtmatr, vmin=0, cmap = "plasma" )
    ax4.set_ylim(0,200)
    ax4.set_ylabel("Frequency in [Hz]")
    ax4.set_xlabel("Time in [s]")
    ax4.set_title("Scaleogram using wavelet")

#################################################################################
################################ Espectrograma ##################################
#################################################################################

def generar_espectrograma(data, fs):
  f, t, Sxx = spectrogram(data, fs=fs, nperseg=128, window='hamming', noverlap=64)  
  return f, t, Sxx

def DibujarEspectrograma(data, fs, f, t, Sxx, titulo):
  plt.figure(figsize=(15, 5))

  # Subplot de la señal de PCG
  plt.subplot(1, 2, 1)
  plt.plot(np.arange(len(data)) / fs, data)
  plt.title('Señal de PCG')
  plt.xlabel('Tiempo (s)')
  plt.ylabel('Amplitud')

  # Subplot del espectrograma
  plt.subplot(1, 2, 2)
  plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
  plt.title('Espectrograma')
  plt.ylabel('Frecuencia (Hz)')
  plt.xlabel('Tiempo (s)')
  plt.colorbar(label='Intensidad (dB)')

  plt.suptitle(titulo)
  plt.tight_layout()
  plt.show()

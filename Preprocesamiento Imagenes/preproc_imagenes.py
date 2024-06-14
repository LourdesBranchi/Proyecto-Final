import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import os


path_normal = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Espectrogramas/Normal'
path_anormal = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Espectrogramas/Anormal'

file_normal = os.listdir(path_normal)
im_normal = []
for i in range(len(file_normal)):
  print(os.path.join(path_normal, file_normal[i]))
  file_n = cv2.imread(os.path.join(path_normal, file_normal[i]))
  im_normal.append(cv2.cvtColor(file_n, cv2.COLOR_BGR2RGB))

file_anormal = os.listdir(path_anormal)
im_anormal = []
for i in range(len(file_anormal)):
  print(os.path.join(path_anormal, file_anormal[i]))
  file_a = cv2.imread(os.path.join(path_anormal, file_anormal[i]))
  im_anormal.append(cv2.cvtColor(file_a, cv2.COLOR_BGR2RGB))

# Función para reducir el ruido
def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Función para ajustar el brillo y el contraste
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Función para aumentar la nitidez
def increase_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

filt_normal = []
filt2_normal = []
filt3_normal = []
for imagen in im_normal:
  filt_normal.append(adjust_brightness_contrast(imagen, brightness=0, contrast=30))
  filt2_normal.append(increase_sharpness(imagen))
  filt3_normal.append(increase_sharpness(adjust_brightness_contrast(imagen, brightness=0, contrast=30)))

filt_anormal = []
filt2_anormal = []
filt3_anormal = []
for imagen in im_anormal:
  filt_anormal.append(adjust_brightness_contrast(imagen, brightness=0, contrast=30))
  filt2_anormal.append(increase_sharpness(imagen))
  filt3_anormal.append(increase_sharpness(adjust_brightness_contrast(imagen, brightness=0, contrast=30)))

# Crear listas para almacenar los histogramas de cada canal
rojo = []
verde = []
azul = []

for imagen in filt2_normal:
  r, g, b = cv2.split(imagen)
  rojo.append(cv2.calcHist([r], [0], None, [256], [0, 256]))
  verde.append(cv2.calcHist([g], [0], None, [256], [0, 256]))
  azul.append(cv2.calcHist([b], [0], None, [256], [0, 256]))

rojo = np.array(rojo)
# Calcular el histograma promedio
average_rojo = np.mean(rojo, axis=0)

verde = np.array(verde)
# Calcular el histograma promedio
average_verde = np.mean(verde, axis=0)

azul = np.array(azul)
# Calcular el histograma promedio
average_azul = np.mean(azul, axis=0)

colors = ('b', 'g', 'r')
plt.figure(figsize=(15, 5))


plt.plot(average_rojo, color='r')
plt.plot(average_azul, color='b')
plt.plot(average_verde, color='g')
plt.xlim([0, 256])

plt.title('Histogramas de los canales de color - Post Filtrado - Normal')
plt.xlabel('Intensidad de los píxeles')
plt.ylabel('Número de píxeles')
plt.show()

# Crear listas para almacenar los histogramas de cada canal
rojo_a = []
verde_a = []
azul_a = []

for imagen in filt2_anormal:
  r, g, b = cv2.split(imagen)
  rojo_a.append(cv2.calcHist([r], [0], None, [256], [0, 256]))
  verde_a.append(cv2.calcHist([g], [0], None, [256], [0, 256]))
  azul_a.append(cv2.calcHist([b], [0], None, [256], [0, 256]))

rojo_a = np.array(rojo_a)
# Calcular el histograma promedio
average_rojo_a = np.mean(rojo_a, axis=0)

verde_a = np.array(verde_a)
# Calcular el histograma promedio
average_verde_a = np.mean(verde_a, axis=0)

azul_a = np.array(azul_a)
# Calcular el histograma promedio
average_azul_a = np.mean(azul_a, axis=0)

colors = ('b', 'g', 'r')
plt.figure(figsize=(15, 5))


plt.plot(average_rojo_a, color='r')
plt.plot(average_azul_a, color='b')
plt.plot(average_verde_a, color='g')
plt.xlim([0, 256])

plt.title('Histogramas de los canales de color - Post Filtrado - Normal')
plt.xlabel('Intensidad de los píxeles')
plt.ylabel('Número de píxeles')
plt.show()

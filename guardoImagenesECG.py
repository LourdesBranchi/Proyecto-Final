import os
import cv2
import json
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.io import savemat

# Cargar las imágenes y etiquetas
data_folder = "C:/Users/lourd/Documents/Proyecto Final/pfc.venv/espectrogramas"
with open('C:/Users/lourd/Documents/Proyecto Final/pfc.venv/latidos.json', 'r') as f:
    latidos = json.load(f)

images = []
labels = []
ecg_data = []
def obtener_etiqueta(latidos, i):
    label = latidos[str(i)]['condicion']
    ecg = latidos[str(i)]['ecg']
    return label, ecg


# Obtener el número total de imágenes
num_images = len(os.listdir(data_folder))

# Iterar sobre los archivos de imagen y mostrar una barra de progreso
for filename in tqdm(os.listdir(data_folder), total=num_images, desc="Cargando imágenes"):
    img = cv2.imread(os.path.join(data_folder, filename))
    if img is not None:
        i = re.search(r'\d+', filename).group()
        images.append(img)
        etiqueta, ecg = obtener_etiqueta(latidos, i)
        ecg_data.append(ecg)
        labels.append(etiqueta)  # Aquí necesitarás una función para obtener la etiqueta de tu nombre de archivo
 
 

# Convertir listas a matrices numpy
images = np.array(images)
labels = np.array(labels)

encoded_labels = []
# Codificación de etiquetas (Label Encoding)
for label in labels:
    if(label == 'Normal'):
        encoded_labels.append(0)
    else:
        encoded_labels.append(1)


# Dividir los datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp, ecg_train, ecg_temp = train_test_split(images, encoded_labels, ecg_data, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test, ecg_val, ecg_test = train_test_split(X_temp, y_temp, ecg_temp, test_size=0.5, random_state=42)


def cont(y):
    normal = 0
    anormal = 0
    for i in y:
        if(i==0):
            normal+=1
        else: 
            anormal+=1
    return normal, anormal

# Imprimir la cantidad de imágenes normales y anormales en cada conjunto
print("Cantidad de imágenes en el conjunto de entrenamiento:")
cont_train = cont(y_train)
print("Normales:", cont_train[0])
print("Anormales:", cont_train[1])
print("")

print("Cantidad de imágenes en el conjunto de validación:")
cont_val = cont(y_val)
print("Normales:", cont_val[0])
print("Anormales:", cont_val[1])
print("")

print("Cantidad de imágenes en el conjunto de prueba:")
cont_test = cont(y_test)
print("Normales:", cont_test[0])
print("Anormales:", cont_test[1])
print("")

# Define los directorios para cada conjunto de datos
train_dir = "C:/Users/lourd/Documents/Proyecto Final/pfc.venv/Scripts/cl_ecg_net/imagenes/train"
test_dir = "C:/Users/lourd/Documents/Proyecto Final/pfc.venv/Scripts/cl_ecg_net/imagenes/test"
val_dir = "C:/Users/lourd/Documents/Proyecto Final/pfc.venv/Scripts/cl_ecg_net/imagenes/val"


def guardar_imagenes_en_directorio(imagenes, etiquetas, latidos, directorio):
    for i, (imagen, etiqueta) in enumerate(zip(imagenes, etiquetas)):
        # Crea un subdirectorio para cada etiqueta
        label_dir = os.path.join(directorio, str(etiqueta))
        os.makedirs(label_dir, exist_ok=True)
        
        # Guarda la imagen en el subdirectorio correspondiente
        cv2.imwrite(os.path.join(label_dir, f"imagen_{i}.jpg"), imagen)
        
        # Obtén los datos de ECG correspondientes a esta imagen del diccionario 'latidos'
        ecg = latidos[i]
        file_mat = f"a{i}_{etiqueta}.mat"
        # Guarda los datos de ECG en un archivo .mat dentro del subdirectorio
        savemat(os.path.join(label_dir, file_mat), {"ecg": ecg})




# Guardar imágenes de entrenamiento
guardar_imagenes_en_directorio(X_train, y_train, ecg_train, train_dir)

# Guardar imágenes de prueba
guardar_imagenes_en_directorio(X_test, y_test, ecg_test, test_dir)

# Guardar imágenes de validación
guardar_imagenes_en_directorio(X_val, y_val, ecg_val, val_dir)


import os

def generar_json_ecg(directorio):
    ecg_list = []
    
    # Recorrer las carpetas train, val y test
    #for carpeta in ["train", "val", "test"]:
    #    carpeta_path = os.path.join(directorio, carpeta)
        
        # Recorrer las subcarpetas 0 y 1 dentro de cada carpeta (normal y anormal)
    for etiqueta in ["0", "1"]:
        etiqueta_path = os.path.join(directorio, etiqueta)
        
        # Recorrer los archivos .mat dentro de la subcarpeta actual
        for archivo in os.listdir(etiqueta_path):
            if archivo.endswith(".mat"):
                # Generar la ruta del archivo .mat
                mat_file_path = os.path.join(etiqueta_path, archivo)
                # Crear el diccionario para este dato ECG
                ecg_dict = {"ecg": mat_file_path, "label": [int(etiqueta)]}
                
                # Agregar el diccionario a la lista
                ecg_list.append(ecg_dict)
    
    # Guardar la lista de diccionarios en un archivo JSON
    with open(os.path.join(directorio, "ecg_data.json"), 'w') as json_file:
        for ecg_dict in ecg_list:
            json.dump(ecg_dict, json_file)
            json_file.write('\n')  # Agregar nueva línea entre cada diccionario


# Generar archivo JSON para datos de entrenamiento
generar_json_ecg(train_dir)

# Generar archivo JSON para datos de prueba
generar_json_ecg(test_dir)

# Generar archivo JSON para datos de validación
generar_json_ecg(val_dir)

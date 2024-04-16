import os
import numpy as np
import cv2
import keras

class Preproc:
    def __init__(self, labels):
        self.classes = sorted(set(labels))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        # No se requiere ningún preprocesamiento específico para las imágenes
        return np.array(x)

    def process_y(self, y):
        # Convertir las etiquetas en enteros y luego en one-hot encoding
        y_int = np.array([self.class_to_int[label] for label in y])
        y_categorical = keras.utils.to_categorical(y_int, num_classes=len(self.classes))
        return y_categorical

    def process_y1(self, y):
        # Convertir las etiquetas en enteros
        y_int = np.array([self.class_to_int[label] for label in y])
        return y_int

def data_generator(batch_size, preproc, images, labels):
    num_examples = len(images)
    examples = list(zip(images, labels))
    np.random.shuffle(examples)
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size] for i in range(0, end, batch_size)]
    
    while True:
        for batch in batches:
            x_bat, y_bat = zip(*batch)
            x_bat = np.array(x_bat)
            x_bat = np.expand_dims(x_bat, axis=-1)  # Agregar dimensión del canal para las imágenes en escala de grises
            y_bat = preproc.process_y(y_bat)
            yield x_bat, y_bat

def data_generator2(preproc, images, labels):
    x = np.array(images)
    x = np.expand_dims(x, axis=-1)  # Agregar dimensión del canal para las imágenes en escala de grises
    y = preproc.process_y(labels)
    return x, y

def load_images(data_dir):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        # Ignorar archivos ocultos (por ejemplo, .DS_Store en macOS)
        if label.startswith('.'):
            continue
        
        label_dir = os.path.join(data_dir, label)
        label_value = int(label)  # Convertir el nombre de la clase a un valor numérico
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(label_value)

    return images, labels

def load_data(data_dir):
    images, labels = load_images(data_dir)
    preproc = Preproc(num_classes=2)  # Para clases binarias (0 y 1)
    return images, labels, preproc

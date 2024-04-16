from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Reshape
from keras import backend as K

def recall(y_true, y_pred):
    # Calculates the recall
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positive / (positive + K.epsilon())

def spe(y_true, y_pred):
    # Calculates the recall
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    negative = K.sum(K.round(K.clip(y_true, 0, 1)))
    true_negative = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_negative / (negative + K.epsilon())

def F1(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    # if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
    #    return 0
    p = spe(y_true, y_pred)
    r = recall(y_true, y_pred)
    fbeta_score = 2 * (p * r) / (p + r + K.epsilon())
    return fbeta_score

def compile(model):
    from keras.optimizers import Adam
    optimizer = Adam(
        learning_rate=0.001,
        clipnorm=1)
    # categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

def build_network(**params):
    # Definir la entrada de la red neuronal
    inputs = Input(shape=params['input_shape'])

    # Definir las capas convolucionales
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    maxpool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxpool2)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    # Aplanar las características para la capa densa
    flatten = Flatten()(conv4)

    # Definir la capa densa
    dense1 = Dense(64, activation='relu')(flatten)

    # Aplicar dropout para regularización
    dropout = Dropout(0.5)(dense1)

    # Definir la capa de salida
    outputs = Dense(2, activation='softmax')(dropout)

    # Crear el modelo
    model = Model(inputs=inputs, outputs=outputs)

    compile(model)

    return model


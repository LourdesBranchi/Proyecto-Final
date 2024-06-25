from function import specificity,cal_auc,LossHistory
import network
import load
import util
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.models import Model
import scipy.io as scio
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score,f1_score
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
    
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

MAX_EPOCHS = 160
batch_size = 32

if __name__ == '__main__':
    params = util.config()
    save_dir = params['save_dir']

    print("Loading training set...")
    train = load.load_dataset(params['train']) # Carga el training set
    
    print("Loading dev set...")
    dev = load.load_dataset(params['dev']) # Carga el validation set
    
    print("Building preprocessor...")
    preproc = load.Preproc(*train) # Inicializamos el preprocesador 
    
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")

    params.update({
        "input_shape": [2048, 1],
        "num_categories": len(preproc.classes)
    })

    # Creamos la red neuronal
    model = network.build_network(**params)

    # Definir una devolución de llamada para guardar los mejores pesos del modelo
    #checkpointer = ModelCheckpoint(filepath=save_dir + 'best_weights.keras', 
    #                               monitor='val_loss', 
    #                               verbose=1, 
    #                               save_best_only=True,
    #                               mode='min')  # Guarda el modelo cuando la pérdida de validación es mínima

    #learning rate reduce strategy
    def scheduler(epoch, lr):
        if epoch % 80 == 0 and epoch != 0:
            lr *= 0.1
            model.optimizer.learning_rate = lr
            print("lr changed to {}".format(lr))
        return lr

    reduce_lr = LearningRateScheduler(scheduler)

    
    #choose best model to save
    checkpointer = keras.callbacks.ModelCheckpoint(
        mode='max',
        monitor='val_accuracy',
        filepath=save_dir + 'best.keras',
        save_best_only=True)

    #variable to save the loss_acc_iter value
    history = LossHistory()

    # Configura TensorBoard
    tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

    # Data generator
    train_gen = load.data_generator(batch_size, preproc, *train)
    dev_gen = load.data_generator(batch_size, preproc, *dev)
    
    #fit the model
    print('cl_ecg_net starts training...')
    model.fit(
        train_gen,
        steps_per_epoch=int(len(train[0]) / batch_size),
        epochs=MAX_EPOCHS,
        validation_data=dev_gen,
        validation_steps=int(len(dev[0]) / batch_size),
        verbose=True,
        callbacks=[checkpointer, reduce_lr, history, tensorboard_callback])

    #save loss_acc_iter
    history.save_result(params['save_dir'] + 'ecg_loss_acc_iter.mat')
    print('Listo. Se guardo en ', params['save_dir'] + 'ecg_loss_acc_iter.mat')
    
    # Graficar las pérdidas de entrenamiento y validación
    history.plot_loss_and_accuracy(params['save_dir'])

    
    #Extract and save deep coding features
    print('Guardando los deep coding features')
    x_train, y_train = load.data_generator2(preproc, *train)
    x, y_t = load.data_generator2(preproc, *dev)
    model.load_weights(save_dir + 'best.keras')
    print('Se guardo en'+ save_dir + 'best_weights.keras')
    new_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    feature_train = new_model.predict(x_train)
    feature_test = new_model.predict(x)
    scio.savemat(save_dir + 'ecg_train.mat', {'x': feature_train, 'y':y_train})
    scio.savemat(save_dir + 'ecg_test.mat', {'x': feature_test, 'y':y_t})
    print('deep coding features of ecg saved')

    #evaluate model
    y_p = model.predict(x)
    y_pred_classes = np.argmax(y_p, axis=1)
    print(confusion_matrix(y_t.argmax(1), y_p.argmax(1)))
    print('sensitivity:', recall_score(y_t.argmax(1), y_p.argmax(1)))
    print('specificity:', specificity(y_t.argmax(1), y_p.argmax(1)))
    print('f1-score:', f1_score(y_t.argmax(1), y_p.argmax(1)))
    print('accuracy:', accuracy_score(y_t.argmax(1), y_p.argmax(1)))
    print('roc:', cal_auc(y_t.argmax(1), y_p[:, 1], save_dir))


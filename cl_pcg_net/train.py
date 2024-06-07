from function import specificity, cal_auc, LossHistory
import network
import load
import util
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Model
import scipy.io as scio
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

MAX_EPOCHS = 160
batch_size = 32

if __name__ == '__main__':
    params = util.config()
    save_dir = params['save_dir']

    print("Loading training set...")
    train_data = load.load_images(params['train'])  # Load images dataset

    print("Loading dev set...")
    dev_data = load.load_images(params['dev'])  # Load dev set

    print("Building preprocessor...")
    preproc = load.Preproc(train_data[1])  # Initialize preprocessor

    print("Training size: " + str(len(train_data[0])) + " examples.")
    print("Dev size: " + str(len(dev_data[0])) + " examples.")

    params.update({
        "input_shape": (107, 340, 3),  # Update input shape according to image dimensions
        "num_categories": len(preproc.classes)  # Update number of categories
    })

    # Create the network model
    model = network.build_network(**params)

    # Define a callback to save the best weights of the model
    checkpointer = ModelCheckpoint(filepath=save_dir + 'best_weights.keras', 
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=True,
                                   mode='min')  # Save the model when the validation loss is at its minimum

    # Learning rate reduction strategy
    def scheduler(epoch, lr):
        if epoch % 80 == 0 and epoch != 0:
            lr *= 0.1
            model.optimizer.learning_rate = lr
            print("lr changed to {}".format(lr))
        return lr
    
    reduce_lr = LearningRateScheduler(scheduler)

    # Choose best model to save
    checkpointer = ModelCheckpoint(
        mode='max',
        monitor='val_accuracy',
        verbose=1,
        filepath=save_dir + 'best.keras',
        save_best_only=True
    )

    # Variable to save the loss_acc_iter value
    history = LossHistory()

    # Data generator
    train_gen = load.data_generator(batch_size, preproc, train_data[0], train_data[1])
    dev_gen = load.data_generator(batch_size, preproc, dev_data[0], dev_data[1])
    
    # Fit the model
    model.fit(
        train_gen,
        steps_per_epoch=int(len(train_data[0]) / batch_size),
        epochs=MAX_EPOCHS,
        validation_data=dev_gen,
        validation_steps=int(len(dev_data[0]) / batch_size),
        verbose=True,
        callbacks=[checkpointer, reduce_lr, history]
    )

    # Save loss_acc_iter
    history.save_result(params['save_dir'] + 'loss_acc_iter.mat')
    print('Done')
    
    # Extract and save deep coding features
    x_train, y_train = load.data_generator2(preproc, train_data[0], train_data[1])
    x_dev, y_dev = load.data_generator2(preproc, dev_data[0], dev_data[1])
    model.load_weights(save_dir + 'best.keras')
    new_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    features_train = new_model.predict(x_train)
    features_dev = new_model.predict(x_dev)
    scio.savemat(save_dir + 'image_train.mat', {'x': features_train, 'y': y_train})
    scio.savemat(save_dir + 'image_dev.mat', {'x': features_dev, 'y': y_dev})
    print('Deep coding features saved')

    # Evaluate the model
    y_pred = model.predict(x_dev)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_dev.argmax(1), y_pred_classes)
    sensitivity = recall_score(y_dev.argmax(1), y_pred_classes)
    specificity_val = specificity(y_dev.argmax(1), y_pred_classes)  # Assuming you have the `specificity` function defined
    f1 = f1_score(y_dev.argmax(1), y_pred_classes)
    accuracy = accuracy_score(y_dev.argmax(1), y_pred_classes)
    roc_auc = cal_auc(y_dev.argmax(1), y_pred[:, 1])  # Assuming you have the `cal_auc` function defined

    # Print evaluation metrics
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity_val)
    print("F1-score:", f1)
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)

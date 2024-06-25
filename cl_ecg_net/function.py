import tensorflow.keras as keras
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_curve,auc

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.acc['batch'].append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.acc['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.loss[loss_type]))


    def save_result(self,path):
        loss_batch = np.array(self.loss['batch'])
        loss_epoch = np.array(self.loss['epoch'])
        acc_batch = np.array(self.acc['batch'])
        acc_epoch = np.array(self.acc['epoch'])
        val_loss_epoch = np.array(self.val_loss['epoch'])
        val_acc_epoch = np.array(self.val_acc['epoch'])
        # Check for None values and handle them
        if loss_batch is None or loss_epoch is None or acc_batch is None or acc_epoch is None:
            raise ValueError("One or more variables to be saved are None.")

        scio.savemat(path, {'loss_batch': loss_batch,'loss_epoch':loss_epoch,'acc_batch':acc_batch,'acc_epoch':acc_epoch,
                            'val_loss_epoch':val_loss_epoch, 'val_acc_epoch':val_acc_epoch})
    def plot_loss_and_accuracy(self, save_dir):
        training_loss = self.loss['epoch']
        validation_loss = self.val_loss['epoch']
        training_accuracy = self.acc['epoch']
        validation_accuracy = self.val_acc['epoch']
        
        # Crear el gráfico de pérdidas
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_dir + 'loss_plot.png')
        plt.close()

        # Crear el gráfico de precisiones
        plt.plot(training_accuracy, label='Training Accuracy')
        plt.plot(validation_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(save_dir + 'accuracy_plot.png')
        plt.close()


def specificity(y_true,y_test):
    TN = 0
    FP = 0
    for i in range(0,y_true.shape[0]):
        if int(y_true[i]) == 0 and int(y_test[i]) == 0:
            TN = TN + 1
        if int(y_true[i]) == 0 and int(y_test[i]) == 1:
            FP = FP + 1
    return  float(TN)/(TN+FP)

def cal_auc(y,y_pre):
    fpr, tpr, thresholds = roc_curve(y, y_pre, pos_label=None, sample_weight=None,drop_intermediate=True)
    auc_area = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_area)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path + 'roc_curve.png')
    plt.close()   
    
    return auc_area

import load
import util

MAX_EPOCHS = 150
batch_size = 32
if __name__ == '__main__':
    params = util.config()
    save_dir = params['save_dir']

    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")
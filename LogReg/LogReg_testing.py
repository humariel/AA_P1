import os
from PIL import Image
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pynput import keyboard
import random

CURRENT_DIR = os.getcwd()
PROCESSED_TRAIN_DIR = CURRENT_DIR + '/../dataset/train'
PROCESSED_TEST_DIR = CURRENT_DIR + '/../dataset/test'
META_DIR = CURRENT_DIR + '/../gtsrb-german-traffic-sign/meta'
DATASET_DIR = CURRENT_DIR + '/../dataset'


def processImage(path):
    image = Image.open(path)
    data = np.asarray(image)
    data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
    return data


def displayTestPred(test_path, prediction_path):
    img_test = mpimg.imread(test_path)
    img_pred = mpimg.imread(prediction_path)

    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img_test)
    plt.title('Test Image')
    f.add_subplot(1,2, 2)
    plt.imshow(img_pred)
    plt.title('Prediction')
    plt.show(block=True)


def getPredictionPath(pred):
    return META_DIR + '/' + str(pred) + '.png' 


def stopLoop(key):
    if key == keyboard.Key.esc:
        print('Stopping... Close current picture to end, if any one is open.')
        return False


def main():
    #load the model
    filename = 'model'
    clf = pickle.load(open(filename, 'rb'))

    test = []
    test_paths = []

    for root, dirs, files in os.walk(PROCESSED_TEST_DIR):
        for name in files:
            path = os.path.join(root,name)
            image = processImage(path)
            test.append(image)
            test_paths.append(path)

    scaler = StandardScaler()
    scaler.fit(test)
    test = scaler.transform(test)

    print('Close the current picture to open the next.')
    print('Press ESC to leave. It will only leave after you\'ve closed the currently open picture.')
    with keyboard.Listener(on_press=stopLoop) as listener:
        for i in range(0,len(test)):
            index = random.randrange(0,len(test))
            pred = clf.predict([test[index]])
            pred_path = getPredictionPath(pred[0])

            displayTestPred(test_paths[index], pred_path)
            if not listener.running:
                break


if __name__ == "__main__":
    main()

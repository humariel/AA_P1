import os
from PIL import Image
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle

CURRENT_DIR = os.getcwd()
PROCESSED_TRAIN_DIR = CURRENT_DIR + '/dataset/train'
PROCESSED_TEST_DIR = CURRENT_DIR + '/dataset/test'
DATASET_DIR = CURRENT_DIR + '/dataset'
TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

def getTrainingSet():
    Xtrain = []
    ytrain = []
    for root, dirs, files in os.walk(PROCESSED_TRAIN_DIR):
        for name in files:
            path = os.path.join(root,name)
            image = processImage(path)
            Xtrain.append(image)
            folder_name = root.split('/')[-1]   #the folder name is the label of the example
            ytrain.append(folder_name)
    return Xtrain, ytrain


def processImage(path):
    image = Image.open(path)
    data = np.asarray(image)
    data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
    return data


def shuffleDataset(X,y):
    if len(X) == len(y):
        perms = np.random.permutation(len(X))
        return X[perms],y[perms]
    else:
        print('X and y must have the same length')


if __name__ == "__main__":
    #collect the training set
    print('Collecting training set. This may take a while ...')
    X, y = getTrainingSet()
    print('Done. Collected {} training examples'.format(len(X)))
    X,y = shuffleDataset(np.asarray(X),np.asarray(y))

    print('Shuffling dataset...')
    trainSize = int(TRAIN_PERCENTAGE * len(X))
    valSize = int(VAL_PERCENTAGE * len(X))
    splitter = [trainSize, trainSize+valSize]
    shuffled_X = np.split(X,splitter)
    shuffled_y = np.split(y,splitter)

    Xtrain = shuffled_X[0]
    Xval = shuffled_X[1]
    Xtest = shuffled_X[2]

    ytrain = shuffled_y[0]
    yval = shuffled_y[1]
    ytest = shuffled_y[2]
    print('Done')

    print('Scaling training set')
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    print('Done')
    print('Scaling vailidation set')
    scaler = StandardScaler()
    scaler.fit(Xval)
    Xval = scaler.transform(Xval)
    print('Done')
    print('Scaling test set')
    scaler = StandardScaler()
    scaler.fit(Xtest)
    Xtest = scaler.transform(Xtest)
    print('Done')

    #save all the sets
    print('Saving all the sets')
    np.save(DATASET_DIR + '/Xtrain', Xtrain)
    np.save(DATASET_DIR + '/Xval', Xval)
    np.save(DATASET_DIR + '/Xtest', Xtest)

    np.save(DATASET_DIR + '/ytrain', ytrain)
    np.save(DATASET_DIR + '/yval', yval)
    np.save(DATASET_DIR + '/ytest', ytest)


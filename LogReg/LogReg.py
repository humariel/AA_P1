import os
from PIL import Image
import numpy as np 
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt

CURRENT_DIR = os.getcwd()
DATASET_DIR = CURRENT_DIR + '/../dataset'


def main():
    #collect the data set
    print('Collecting Xtrain')
    Xtrain = np.load(DATASET_DIR + '/Xtrain.npy')
    print('Collecting Xtest')
    Xtest = np.load(DATASET_DIR + '/Xtest.npy')

    print('Collecting ytrain')
    ytrain = np.load(DATASET_DIR + '/ytrain.npy')
    print('Collecting yval')
    ytest = np.load(DATASET_DIR + '/ytest.npy')

    #use only 10% of data
    Xtrain = Xtrain[0:int(len(Xtrain)/10)]
    ytrain = ytrain[0:int(len(ytrain)/10)]

    Xtest = Xtest[0:int(len(Xtest)/10)]
    ytest = ytest[0:int(len(ytest)/10)]

    #create the classifier and fit to the data
    print('Creating the model...')
    clf = LogisticRegression(verbose=True, max_iter=5000, C=0.3)
    clf.fit(Xtrain, ytrain)
    print('Done')
    result = clf.score(Xtest, ytest)
    print(result)
    #save the classifier
    filename = 'model'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    main()



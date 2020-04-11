import os
from PIL import Image
import numpy as np 
from sklearn.neural_network import MLPClassifier
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


    #create the classifier and fit to the data
    print('Creating the model...')
    k = 43                  #number of classes
    neurons = 100  #neurons in hidden layer
    clf = MLPClassifier(solver='sgd',activation='relu', alpha=0.003, learning_rate_init=0.01, hidden_layer_sizes=(neurons, k), max_iter=200, momentum=0.9, nesterovs_momentum=True, verbose=True)
    print('Fitting the model. This will take a while...')
    clf.fit(Xtrain, ytrain)

    res = clf.score(Xtest,ytest)
    print(res)

    plt.plot(clf.loss_curve_)
    plt.show()

    #save the classifier
    filename = 'model'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    main()



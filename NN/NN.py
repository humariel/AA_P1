import os
from PIL import Image
import numpy as np 
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

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
    neurons = 350  #neurons in hidden layer
    clf = MLPClassifier(solver='sgd',activation='relu', alpha=0.03, learning_rate_init=0.003, hidden_layer_sizes=(neurons), max_iter=5000, momentum=0.9, nesterovs_momentum=True, verbose=True)
    print('Fitting the model. This will take a while...')
    clf.fit(Xtrain, ytrain)

    res = clf.score(Xtest,ytest)
    print(res)

    # preds = clf.predict(Xtest)
    # f1 = f1_score(ytest, preds, average='weighted')
    # precision = precision_score(ytest, preds, average='weighted')
    # recall = recall_score(ytest, preds, average='weighted')
    # print(f1)
    # print(recall)
    # print(precision)

    plt.plot(clf.loss_curve_)
    plt.show()

    #save the classifier
    filename = 'model'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    main()



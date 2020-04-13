import os
from PIL import Image
import numpy as np 
from sklearn.svm import LinearSVC, SVC
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
import time

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
    clf = SVC(C=0.003, verbose=True, kernel='linear')
    start = time.time()
    clf.fit(Xtrain, ytrain)
    end = time.time()
    print('Done')

    preds = clf.predict(Xtest)
    acc = accuracy_score(ytest, preds)
    f1 = f1_score(ytest, preds, average='weighted')
    precision = precision_score(ytest, preds, average='weighted')
    recall = recall_score(ytest, preds, average='weighted')
    print('Accuracy:' + str(acc))
    print('F1:'+str(f1))
    print('Recall:'+str(recall))
    print('Precision:'+str(precision))
    print('Took {} seconds'.format(str(end-start)))


    #save the classifier
    filename = 'model'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    main()



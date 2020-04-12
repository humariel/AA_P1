import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import sys
import argparse


CURRENT_DIR = os.getcwd()
DATASET_DIR = CURRENT_DIR + '/../dataset'


def run_optimizer():
    #load the model
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

    params = {
        'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 9999]   
    }

    scores = {
        'f1_' : 'f1_weighted',
        'loss': 'neg_log_loss',
        'acc': 'accuracy'
    }

    model = SVC(kernel='linear', probability=True)

    optimizer = GridSearchCV(estimator=model, param_grid=params, cv=3, verbose=10, scoring=scores, refit='acc')
    print('Fitting optimizer...')
    optimizer.fit(Xtrain, ytrain)
    
    # Best parameter set
    print('Best parameters found:\n', optimizer.best_params_)

    # All results
    joblib.dump(optimizer.best_params_, 'best_params')
    joblib.dump(optimizer.cv_results_, 'optimizer_results')


def plot_data():
    #extract data
    results = joblib.load('optimizer_results')
    
    params = results['params']
    time_to_train = zip(params, results['mean_fit_time'])

    mean_f1 = list(zip(params, results['mean_test_f1_']))
    std_f1 = list(zip(params, results['std_test_f1_']))
    ranking_f1 = results['rank_test_f1_']
    mean_acc =  list(zip(params, results['mean_test_acc']))
    std_acc =  list(zip(params, results['std_test_acc']))
    ranking_acc = results['rank_test_acc']

    loss = list(zip(params, results['mean_test_loss']))

    create_graph(mean_f1, mean_acc, loss)


def create_graph(dict_f1, dict_acc, dict_loss):
    width = 0.2

    create_barh(dict_f1, offset=-width, text_offset=-width, color='b', label='f1_score(in %)', va='center')
    create_barh(dict_acc, text_offset=-width/2.0, color='r', label='accuracy(in %)', va='bottom')
    create_barh(dict_loss, offset=width, text_offset=width/2.0, color='g', label='log_loss', va='bottom')

    plt.title('SVC analysis')
    plt.xlabel('%(f1_score and accuracy)/measured loss(log_loss)')
    plt.legend()
    plt.show()


def create_barh(dict_, color, label, text_offset, offset=0, va='center'):
    plt.barh([x+offset for x in range(0,len(dict_))], [abs(x[1]) for x in dict_], 
                    tick_label=['C={}'.format(x[0]['C']) for x in dict_], color=color, height=0.2, label=label)
    for i,v in enumerate([x[1] for x in dict_]):
        plt.text(abs(v),i+text_offset, " "+str('{:.4f}'.format(abs(v))), va=va)


def main(args):
    if args.option[0] == 0:
        print('Running optimizer')
        run_optimizer()
    elif args.option[0] == 1:
        print('Plotting optimizer results')
        plot_data()
    else:
        print('Invalid value for --options.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NN optimizer and analysis.')
    parser.add_argument('--option', type=int, nargs=1, required=True, choices=[0,1], help='0 will run the optimizer, 1 will show it\'s results')
    args = parser.parse_args()
    main(args)

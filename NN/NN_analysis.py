import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
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

    k = 43          #number of classes

    params = {
        'hidden_layer_sizes' : [(100,k)],
        'activation' : ['relu'],
        'solver' : ['sgd', 'adam'],
        'alpha' : [0.0001, 0.003, 0.001],
        'learning_rate_init' : [0.001, 0.03, 0.01]
    }

    scores = {
        'f1_' : 'f1_weighted',
        'loss': 'neg_log_loss',
        'acc': 'accuracy'
    }

    model = MLPClassifier(max_iter=200)

    optimizer = GridSearchCV(estimator=model, param_grid=params ,cv=3, verbose=10, scoring=scores, refit='acc')
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

    mean_f1_alpha_1 = [x for x in mean_f1 if x[0]['alpha'] == 0.0001]
    mean_f1_alpha_2 = [x for x in mean_f1 if x[0]['alpha'] == 0.003]
    mean_f1_alpha_3 = [x for x in mean_f1 if x[0]['alpha'] == 0.001]

    mean_acc_alpha_1 = [x for x in mean_acc if x[0]['alpha'] == 0.0001]
    mean_acc_alpha_2 = [x for x in mean_acc if x[0]['alpha'] == 0.003]
    mean_acc_alpha_3 = [x for x in mean_acc if x[0]['alpha'] == 0.001]

    create_barh(mean_f1_alpha_1, mean_acc_alpha_1)
    create_barh(mean_f1_alpha_2, mean_acc_alpha_2)
    create_barh(mean_f1_alpha_3, mean_acc_alpha_3)


def create_barh(dict_f1, dict_acc):
    #f1 bar
    plt.barh([x-0.2 for x in range(0,len(dict_f1))], [x[1] for x in dict_f1], 
                    tick_label=['learn_rate={}&{}'.format(x[0]['learning_rate_init'],x[0]['solver']) for x in dict_f1], color='b', height=0.2)
    for i,v in enumerate([x[1] for x in dict_f1]):
        plt.text(v,i, " "+str('{:.4f}'.format(v)), va='bottom')
    #acc bar
    plt.barh(range(0,len(dict_acc)), [x[1] for x in dict_acc], 
                    tick_label=['learn_rate={}&{}'.format(x[0]['learning_rate_init'],x[0]['solver']) for x in dict_acc], color='r', height=0.2)
    for i,v in enumerate([x[1] for x in dict_acc]):
        plt.text(v,i, " "+str('{:.4f}'.format(v)), va='top')

    plt.title('With alpha={}'.format(dict_f1[0][0]['alpha']))

    plt.show()


def plot_convergence_curves():
    params = [{'momentum': 0},
        {'momentum': 0.9,
        'nesterovs_momentum': False},
        {'momentum': 0.9,
        'nesterovs_momentum': True}
    ]
    plot_params = [{'c': 'red'}, {'c':'blue'}, {'c':'green'}]
    labels = ['no momentum', 'momentum', 'nesterovs momentum']

    #collect the data set
    print('Collecting Xtrain')
    Xtrain = np.load(DATASET_DIR + '/Xtrain.npy')
    print('Collecting Xval')
    Xval = np.load(DATASET_DIR + '/Xval.npy')
    print('Collecting Xtest')
    Xtest = np.load(DATASET_DIR + '/Xtest.npy')

    print('Collecting ytrain')
    ytrain = np.load(DATASET_DIR + '/ytrain.npy')
    print('Collecting yval')
    yval = np.load(DATASET_DIR + '/yval.npy')
    print('Collecting yval')
    ytest = np.load(DATASET_DIR + '/ytest.npy')


    #create the classifier and fit to the data
    k = 43                  #number of classes
    neurons = 100  #neurons in hidden layer

    loss_curves = []

    for conf in params:
        print('Creating the model...')
        clf = MLPClassifier(solver='sgd',activation='relu', alpha=0.003, learning_rate_init=0.01, hidden_layer_sizes=(neurons, k), max_iter=200, verbose=True, **conf)
        print('Fitting the model. This will take a while...')
        clf.fit(Xtrain, ytrain)
        loss_curves.append(clf.loss_curve_)
    
    for curve, label, conf in zip(loss_curves, labels, plot_params):
        plt.plot(curve, label=label, **conf)

    plt.legend()
    plt.show()

def main(args):
    if args.option[0] == 0:
        print('Running optimizer')
        run_optimizer()
    elif args.option[0] == 1:
        print('Plotting optimizer results')
        plot_data()
    elif args.option[0] == 2:
        print('Calculating and Plotting convergence curves')
        plot_convergence_curves()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NN optimizer and analysis.')
    parser.add_argument('--option', type=int, nargs=1, required=True, choices=[0,1,2], help='0 will run the optimizer, 1 will show it\'s results \
        and 2 will get the convergence curves for the best configuration with different momentums')
    args = parser.parse_args()
    main(args)

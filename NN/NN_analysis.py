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

    #use only 10% of data
    Xtrain = Xtrain[0:int(len(Xtrain)/10)]
    ytrain = ytrain[0:int(len(ytrain)/10)]

    Xtest = Xtest[0:int(len(Xtest)/10)]
    ytest = ytest[0:int(len(ytest)/10)]

    params = {
        'hidden_layer_sizes' : [(100)],
        'alpha' : [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        'learning_rate_init' : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    }

    scores = {
        'f1_' : 'f1_weighted',
        'loss': 'neg_log_loss',
        'acc': 'accuracy'
    }

    #model = MLPClassifier(max_iter=200)
    model = MLPClassifier(solver='sgd',activation='logistic', max_iter=5000, momentum=0.9, nesterovs_momentum=True)

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

    loss = list(zip(params, results['mean_test_loss']))

    mean_f1_1 = [x for x in mean_f1 if x[0]['learning_rate_init'] == 0.001]
    mean_f1_2 = [x for x in mean_f1 if x[0]['learning_rate_init'] == 0.003]
    mean_f1_3 = [x for x in mean_f1 if x[0]['learning_rate_init'] == 0.01]
    mean_f1_4 = [x for x in mean_f1 if x[0]['learning_rate_init'] == 0.03]

    mean_acc_1 = [x for x in mean_acc if x[0]['learning_rate_init'] == 0.001]
    mean_acc_2 = [x for x in mean_acc if x[0]['learning_rate_init'] == 0.003]
    mean_acc_3 = [x for x in mean_acc if x[0]['learning_rate_init'] == 0.01]
    mean_acc_4 = [x for x in mean_acc if x[0]['learning_rate_init'] == 0.03]

    loss_1 = [x for x in loss if x[0]['learning_rate_init'] == 0.001]
    loss_2 = [x for x in loss if x[0]['learning_rate_init'] == 0.003]
    loss_3 = [x for x in loss if x[0]['learning_rate_init'] == 0.01]
    loss_4 = [x for x in loss if x[0]['learning_rate_init'] == 0.03]

    create_graph(mean_f1_1, mean_acc_1, loss_1)
    create_graph(mean_f1_2, mean_acc_2, loss_2)
    create_graph(mean_f1_3, mean_acc_3, loss_3)
    create_graph(mean_f1_4, mean_acc_4, loss_4)


def create_graph(dict_f1, dict_acc, dict_loss):
    width = 0.2

    create_barh(dict_f1, offset=-width, text_offset=-width, color='b', label='f1_score', va='center')
    create_barh(dict_acc, text_offset=-width/2.0, color='r', label='accuracy', va='bottom')
    create_barh(dict_loss, offset=width, text_offset=width/2.0, color='g', label='log_loss', va='bottom')

    plt.title('With learning rate={}'.format(dict_f1[0][0]['learning_rate_init']))
    plt.legend()
    plt.show()


def create_barh(dict_, color, label, text_offset, offset=0, va='center'):
    plt.barh([x+offset for x in range(0,len(dict_))], [abs(x[1]) for x in dict_], 
                    tick_label=['alpha={}'.format(x[0]['alpha']) for x in dict_], color=color, height=0.2, label=label)
    for i,v in enumerate([x[1] for x in dict_]):
        plt.text(abs(v),i+text_offset, " "+str('{:.4f}'.format(abs(v))), va=va)


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
    k = 43                  #number of classes
    neurons = 100  #neurons in hidden layer

    loss_curves = []

    for conf in params:
        print('Creating the model...')
        clf = MLPClassifier(solver='sgd',activation='logistic', alpha=0.03, learning_rate_init=0.003, hidden_layer_sizes=(neurons), max_iter=5000, verbose=True, **conf)
        print('Fitting the model. This will take a while...')
        clf.fit(Xtrain, ytrain)
        loss_curves.append(clf.loss_curve_)
    
    for curve, label, conf in zip(loss_curves, labels, plot_params):
        plt.plot(curve, label=label, **conf)

    plt.title('Iterations to converge with different momentum')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
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
    else:
        print('Invalid value for --options.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NN optimizer and analysis.')
    parser.add_argument('--option', type=int, nargs=1, required=True, choices=[0,1,2], help='0 will run the optimizer, 1 will show it\'s results \
        and 2 will get the convergence curves for the best configuration with different momentums')
    args = parser.parse_args()
    main(args)

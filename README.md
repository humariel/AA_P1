<h1>AA Project1 </h1>

How to set up:

1. Download the dataset here: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign and place it on this folder.

2. Make sure you have the following things installed in python3:
	1. sklearn
	2. PIL
	3. matplotlib
	4. numpy
	5. pickle
	6. numpy
	7. joblib
	8. pynput

3. Run:
	1. createEmptyDirectories.py
	2. preprocessImages.py
	3. shuffleDivideData.py

4. In the NN folder:
	1. The NN.py script trains a model with the best parameters found from a small amount of possibilities.
	2. The NN_testing.py shows the trained model making predictions.It creates plots with the prediction and actual image. Closig a plot opens the next one. Press ESC to stop the process(you'll have to close the current image).
	3. The NN_analysis.py can (depending on input, run with -h for help) either run the optimization process or show it's results.

5. Similar steps as in 4 for the Logistic Regression and the SVM

NOTE: In the folders mentioned in steps 4 and 5 reside the results of the optimization for each model, meaning there's no need to run it(this process takes a very long time). In those folders you can also find a file that contains the trained model, EXCEPT FOR THE SVM MODEL. The reason why the SVM model isn't there is because it is too heavy to then send it. Fortunately, the SVM is fast to train, and so, if you want to use the SVM_testing.py script, you should first run SVM.py (should only take 1-2 mins).

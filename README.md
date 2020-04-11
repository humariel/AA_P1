<h1>AA Project1 </h1>

How to set up:

1. Download the dataset here: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

2. Make sure you have the following things installed in python3 (in my case python3.7):
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

4. Move to the NN folder:
	1. The NN.py script trains a model with the best parameters found from a small amount of possibilities.
	2. The NN_testing.py shows the trained model making predictions.
	3. The NN_analysis.py can (depending on input, run it with -h for help)
5. Similar steps as in 4 for the Logistic Regression and the SVM

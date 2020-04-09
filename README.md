AA Project1

How to set up:

1. Download the dataset here: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

2. Make sure you have the following things installed in python3 (in my case python3.7):
	-> sklearn
	-> PIL
	-> matplotlib
	-> numpy
	-> pickle
	-> numpy
	-> joblib
	-> pynput

3. Run:
	-> createEmptyDirectories.py
	-> preprocessImages.py
	-> shuffleDivideData.py

4. Move to the NN folder:
	-> The NN.py script trains a model with the best parameters found from a small amount of possibilities.
	-> The NN_testing.py shows the trained model making predictions.
	-> The NN_analysis.py can (depending on input, run it with -h for help)

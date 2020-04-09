import os

os.mkdir(os.getcwd() + '/dataset')
os.mkdir(os.getcwd() + '/dataset/train')
os.mkdir(os.getcwd() + '/dataset/test')

for i in range(0,43):
    os.mkdir(os.getcwd() + '/dataset/train/' + str(i))

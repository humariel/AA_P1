#adapted from
#https://www.kaggle.com/gauss256/preprocess-images
#credits to gauss256

import glob
from multiprocessing import Process
import os
import re

import numpy as np
import PIL
from PIL import Image


# Processing parameters
SIZE = 50                           # 2500 features/pixels per image (50*50)
CURRENT_DIR = os.getcwd()
TEST_DIR = CURRENT_DIR + '/gtsrb-german-traffic-sign/test'
TRAIN_DIR = CURRENT_DIR + '/gtsrb-german-traffic-sign/train' 
PROCESSED_TRAIN_DIR = CURRENT_DIR + '/dataset/train'
PROCESSED_TEST_DIR = CURRENT_DIR + '/dataset/test'

def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_image(path, out_dir):
    """
    Preprocess images

    Reads image in path, and writes to out_dir

    """
    img = Image.open(path)
    img_res = resize_image(img, SIZE)
    basename = os.path.basename(path)
    path_out = os.path.join(out_dir, basename)
    img_res.save(path_out)


def main():
    #preprocess taining set
    print('Processing the train images')
    for root, dirs, files in os.walk(TRAIN_DIR):
        for name in files:
            path = os.path.join(root, name)
            folder_name = root.split('/')[-1]
            output_dir = PROCESSED_TRAIN_DIR + '/' + folder_name + '/'
            new_image = prep_image(path,output_dir)

    #preprocess test set
    #before running temporarily move the csv file in the test folder to another folder (just to avoid the necessity of the 'if', for performance)
    print('Processing the test images')
    for root, dirs, files in os.walk(TEST_DIR):
        for name in files:
            path = os.path.join(root, name)
            output_dir = PROCESSED_TEST_DIR + '/'
            new_image = prep_image(path,output_dir)


if __name__ == "__main__":
    main()

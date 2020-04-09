#https://www.kaggle.com/gauss256/preprocess-images

"""
Preprocess image data for Dogs vs. Cats competition
https://www.kaggle.com/gauss256

DESCRIPTION

Most solutions for this competition will need to deal with the fact that the
training and test images are of varying sizes and aspect ratios, and were taken
under a variety of lighting conditions. It is typical in cases like this to
preprocess the images to make them more uniform.

By default this script will normalize the image luminance and resize to a
square image of side length 224. The resizing preserves the aspect ratio and
adds gray bars as necessary to make them square. The resulting images are
stored in a folder named data224.

The location of the input and output files, and the size of the images, can be
controlled through the processing parameters below.

INSTALLATION

This script has been tested on Ubuntu 14.04 using the Anaconda distribution for
Python 3.5. The only additional requirement is for the pillow library for image
manipulation which can be installed via:
    conda install pillow
"""
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

def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

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
    img_nrm = norm_image(img)
    img_res = resize_image(img_nrm, SIZE)
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

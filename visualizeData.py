from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CURRENT_DIR = os.getcwd()
PROCESSED_TEST_DIR = CURRENT_DIR + '/dataset/test'

test_paths = []

for root, dirs, files in os.walk(PROCESSED_TEST_DIR):
    for name in files:
        path = os.path.join(root,name)
        test_paths.append(path)

cols = 10
rows = 10
fig=plt.figure(figsize=(10, 10))

for i in range(1, cols*rows +1):
    img = mpimg.imread(test_paths[i])
    fig.add_subplot(rows, cols, i)
    plt.imshow(img)
plt.show()
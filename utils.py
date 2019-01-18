# Includes
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# plot images
def plot_images(images, titles=None, cols=3, fontsize=12):

    n_imgs = len(images)

    if images is None or n_imgs < 1:
        print("No images to display.")
        return

    img_h, img_w = images[0].shape[:2]
    rows = math.ceil(n_imgs / cols)
    width = 21  # 15
    row_height = math.ceil((width/cols)*(img_h/img_w))  # they are 1280*720

    plt.figure(1, figsize=(width, row_height * rows))

    for i, image in enumerate(images):
        if len(image.shape) > 2:
            cmap = None
        else:
            cmap = 'gray'
        title = ""
        if titles is not None and i < len(titles):
            title = titles[i]
        plt.subplot(rows, cols, i+1)
        plt.title(title, fontsize=fontsize)
        plt.imshow(image, cmap=cmap)

    plt.tight_layout()
    plt.show()

# read a csv file to dict
def csv_to_dict(csv_file, print_keys=False):
    
    with open(csv_file, mode='r') as infile:
        csv_reader = csv.reader(infile)
        # skip the header 'ClassId, SignName'
        _ = next(csv_reader)
        # read the sign id and names
        ret_dict = {rows[0]:rows[1] for rows in csv_reader}
    
    if print_keys is True:
        print('Dict keys:', ret_dict.keys())
    
    return ret_dict

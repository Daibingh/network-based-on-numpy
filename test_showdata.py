# -*- coding: utf-8 -*-

import mnist_loader
# import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def showimg(x):
    pass

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    # img = cv2.imread('lena.png')
    # cv2.namedWindow('lena')
    # cv2.imshow('lena', img)
    # cv2.waitKey(0)
    print(len(training_data))
    i = random.randint(0, 5000)
    num = 200
    plt.figure('Image')
    for ii in range(1, num+1):
        img = training_data[i+ii-1][0].reshape((28, 28))
        # cv2.namedWindow('Image')
        img *= 255
        img_2 = img.astype(np.uint8)
        plt.subplot(10, 20, ii)
        plt.imshow(img_2, cmap='gray')
        plt.axis('off')
    plt.show()

    # img = training_data[5][0].reshape((28, 28))
    # # cv2.namedWindow('Image')
    # img *= 255
    # img_2 = img.astype(np.uint8)
    # plt.figure('Image')
    # plt.imshow(img_2,cmap='gray')
    # plt.axis('off')
    # plt.show()
    # cv2.imshow('Image', img_2)
    # cv2.waitKey(0)
    print('-------------------------------')
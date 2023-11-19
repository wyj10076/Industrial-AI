import cv2
import numpy as np

CELL_SIZE = 20
NCLASSES = 10
TRAIN_RATIO = 0.8

digits_img = cv2.imread("..data/digists.png", 0)
digits = [np.hsplit(r, digits_img.shape[1] // CELL_SIZE)
           for r in np.vsplit(digits_img, digits_img.shape[0] // CELL_SIZE)]

digits = np.array(digits).reshape(-1, CELL_SIZE, CELL_SIZE)
nsamples = digits.shape[0]
labels = np.repeat(np.arange())
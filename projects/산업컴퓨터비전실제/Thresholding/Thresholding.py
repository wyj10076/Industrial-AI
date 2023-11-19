import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("data/Lena.png", 0)

thr, mask = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY)
print("Threshold used:", thr)

adapt_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 10)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.axis("off")
plt.title("original")
plt.imshow(image, cmap="gray")
plt.subplot(132)
plt.axis("off")
plt.title("binary threshold")
plt.imshow(mask, cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("adaptive threshold")
plt.imshow(adapt_mask, cmap="gray")
plt.tight_layout()
plt.show()
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(img, aabbs):
    img = ((img + 0.5) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for aabb in aabbs:
        aabb = aabb.enlarge_to_int_grid().as_type(int)
        cv2.rectangle(img, (aabb.xmin, aabb.ymin), (aabb.xmax, aabb.ymax), (255, 0, 255), 2)

    return img


def visualize_and_plot(img, aabbs):
    plt.imshow(img, cmap='gray')
    for aabb in aabbs:
        plt.plot([aabb.xmin, aabb.xmin, aabb.xmax, aabb.xmax, aabb.xmin],
                 [aabb.ymin, aabb.ymax, aabb.ymax, aabb.ymin, aabb.ymin])

    plt.show()

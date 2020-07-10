import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt

def imshow(inp, title=None , mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]):
    """Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std  = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
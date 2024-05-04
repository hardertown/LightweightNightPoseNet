import os
import json
from torch import optim
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(log_dir, train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(train_loss_list, c='r', label='train loss', linewidth=2)
    plt.plot(val_loss_list, c='b', label='val loss', linewidth=2)
    plt.legend(loc='best')
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.yscale("log")
    plt.savefig(os.path.join(log_dir, 'train_val_loss.jpg'), dpi=600, bbox_inches='tight')
import numpy as np
import matplotlib.pyplot as plt

########################   Visualizing    ###########################################
def visualize_uniq_classes(dataset):
    unq, unq_inv, unq_cnt = np.unique(dataset.labels, return_inverse=True, return_counts=True)
    unq, unq_inv = np.unique(dataset.labels, return_inverse=True)
    class_groups = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    
    for label in class_groups:
        plt.plot(dataset.data[label[0]])
        plt.show()
        plt.close()    
    

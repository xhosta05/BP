# import packages
import re
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

import tensorflow as tf 
import argparse

# .py files
import nets as nets
from tf_dataset import *
from data_processing import *
from utils import *

# TODO visualizing save pictures

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', default="conv_1l")
    parser.add_argument('-d', '--data-path', help='Path to training csv files', required=True)
    parser.add_argument('--visualize', default=False, type=bool) 
    parser.add_argument('-b', '--batch-size', default=256, type=int) 
    parser.add_argument('-e', '--epochs', default=8, type=int) 
    parser.add_argument('-s', '--model-save-path', default="./", type=str) 

    args = parser.parse_args()
    return args
    
def test(test_data,test_labels,model):
    ########################   Testing    ###########################################

    test_pred_arg = np.argmax(model.predict(test_data), axis=1)
    test_labels = np.squeeze(test_labels)

    correct = test_pred_arg == test_labels
    acc = correct.sum()/len(test_labels)
    return acc

def main():
    print("START")
    args = parse_arguments()

    dataset = csvs_array2Dataset(args.data_path)
    dataset_all = np.concatenate((dataset.labels[:,None], dataset.data), axis=1)

    np.random.shuffle(dataset_all)
    split_idx= int(dataset_all.shape[0]*0.2)
    num_classes=len(np.unique(dataset.labels))

    train_dataset = dataset_all[split_idx:]
    test_dataset = dataset_all[:split_idx]

    train_labels =train_dataset[: , :1].astype(int)
    
    Targets = one_hot_encode(train_labels)
    Dataset  = train_dataset[:, 1:] 
    
    The_Model = nets.get_model(args.name, Dataset[0].shape, num_classes)
    
    ########################   Training    ###########################################
    The_Model.fit(Dataset, Targets, batch_size = args.batch_size, epochs = args.epochs, shuffle = True, verbose = 1)
    model_name_var= "b"+str(args.batch_size)+"_e"+str(args.epochs)
    The_Model.save(os.path.join(args.model_save_path, args.name,model_name_var))  


    test_data  = test_dataset[:, 1:] 
    test_labels  = test_dataset[:, :1].astype(int) 

    acc=test(test_data,test_labels,The_Model)
    print("Test accuracy: ",acc*100,"%")
    return 0
    
if __name__ == "__main__":
    main()

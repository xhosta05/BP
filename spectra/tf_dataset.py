import os
import numpy as np
import re

from data_processing import *
import utils

# class Dataset(Dataset):
# 	def __init__(self, features, labels, transform=None, target_transform=None):
# 		#self.data_points_count= data_points_count
# 		#self.data = np.zeros((0, data_points_count))   
# 		#self.labels = np.zeros((0))
# 		#self.classes = {}
# 		self.features = features
# 		self.labels = labels

# 		# self.transform = transform
# 		# self.target_transform = target_transform

# 	def __len__(self):
# 		return len(self.features)
	
# 	"""
# 	def add_items(self, data, labels):
# 		data = data[:, 0:self.data_points_count]
# 		self.data = np.concatenate((self.data, data))
# 		self.labels = np.concatenate((self.labels, labels))

# 	def add_class(self, label, meas_class):
# 		self.classes[label] =   meas_class
# 	"""
	
# 	def add_item(features,label):
# 		self.features.append(feature)
# 		self.labels.append(label)

# 	def __getitem__(self, idx):
# 		feature =  self.features[idx]
# 		label =  self.labels[idx]
# 		return feature, label
class Dataset:
#     normalize when loADING
  def __init__(self, data_points_count= 2047):    
    self.data_points_count= data_points_count
    self.data = np.zeros((0, data_points_count))   
    self.labels = np.zeros((0))
    self.classes = {}
    
  def add_items(self, data, labels):
    data = data[:, 0:self.data_points_count]
    self.data = np.concatenate((self.data, data))
    self.labels = np.concatenate((self.labels, labels))
    
  def add_class(self, label, meas_class):
    self.classes[label] =   meas_class

def create_dataset(measurments,label, label_class):
  dataset=Dataset(len(measurments[0]))
  measurments = [i for i in measurments if i is not None]
  
  dataset.add_class(label, label_class)
  
  for measurment in measurments:
      dataset.add_items(np.array(measurment[None,:]),np.array([label]))
        
#   print(dataset.get_data_shape())
  return dataset
  
def data_to_loader(dataset,dtype,batch_size=16):
    # features=np.array([baseline_als(i.data, 1e6) for i in dataset])
    features=np.array([i.data for i in dataset])
#     features=normalized(features)
    features=torch.tensor(features, dtype=dtype)

    target=np.array([i.label for i in dataset])
    target=torch.tensor(target, dtype=torch.int64)

    tensorDataset = data_utils.TensorDataset(features.to(device), target)
    loader = data_utils.DataLoader(tensorDataset, batch_size=batch_size, shuffle=False)

    return loader,features,target
  
def csvs_array2Dataset(train_data_path):
	"""
	Converts multiple cvs files to dataset class with their contents.

	@param train_data_path: path to csv files (doesn't look in subfolders)
	@return: Dataset class with measurements  
	"""
	datasets=[]
	for root, dirs, files in os.walk(train_data_path):
		for i,name in enumerate(files):
		    if re.findall('.csv', name):
		        # print(i,' ',name)
		        data_arr=np.loadtxt(os.path.join(root, name), delimiter=',')
		        data_arr=data_normalize(data_arr) 

		        dataset= create_dataset(data_arr,i, name)
	#             print(dataset.data.shape)
		        datasets.append(dataset)
		break   #prevent descending into subfolders

	dataset=Dataset()
	for ds in datasets:
		dataset.add_items(ds.data, ds.labels)
		for key in ds.classes:
			dataset.add_class(key, ds.classes[key])
	return dataset

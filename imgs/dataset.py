import numpy as np
import os

from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
	def __init__(self, features, labels, transform=None, target_transform=None):
		#self.data_points_count= data_points_count
		#self.data = np.zeros((0, data_points_count))   
		#self.labels = np.zeros((0))
		#self.classes = {}
		self.features = features
		self.labels = labels

		# self.transform = transform
		# self.target_transform = target_transform

	def __len__(self):
		return len(self.features)
	
	"""
	def add_items(self, data, labels):
		data = data[:, 0:self.data_points_count]
		self.data = np.concatenate((self.data, data))
		self.labels = np.concatenate((self.labels, labels))

	def add_class(self, label, meas_class):
		self.classes[label] =   meas_class
	"""
	
	def add_item(features,label):
		self.features.append(feature)
		self.labels.append(label)

	def __getitem__(self, idx):
		feature =  self.features[idx]
		label =  self.labels[idx]
		return feature, label

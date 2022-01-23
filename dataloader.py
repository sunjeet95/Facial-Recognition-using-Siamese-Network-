# Jai Shri Ram | Om Hum Hanumate Namah
# 5th November, 2021
# Sunjeet Jena
# This is the dataloader.py file for creating the customized dataset loader for the model

#=================Importing the required libraries=====================================
import numpy as np
from PIL import Image
import torch
import math
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import json
import pandas
import torchvision.models as models
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
#======================================================================================

class faceDetection(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, samples, root_dir, transform=None):
		"""
		Args:
			"samples" is the pandas dataframe 
			"root_dir" is root directory where the images are located
			"transform" is tranform to be applied on the image
		"""
		self.samples = samples
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# Read the pair of images and their corresponding label
		img_1 = os.path.join(self.root_dir, self.samples.iloc[idx, 0])
		img_2 = os.path.join(self.root_dir, self.samples.iloc[idx, 1])
		pair_label = self.samples.iloc[idx, 2]

		image1 = Image.open(img_1)
		image2 = Image.open(img_2)

		if self.transform:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		
		return image1, image2, pair_label



def dataLoader(trainSamples, valSamples, data_dir,batch_size, input_shape):
	
	print("Initializing the DataLoaders...........")

	# Creating the transform function
	transformImg = transforms.Compose([transforms.Resize((input_shape, input_shape)),
                              	  	transforms.ToTensor(),
                              	  	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


	# Create the object for train loader
	trainLoader = faceDetection(trainSamples, data_dir, transformImg)

	# Create the object for validation loader
	valLoader = faceDetection(valSamples, data_dir, transformImg)

	# Create a dictionary to store both the training and validation dataloaders

	dataLoader = {'train': DataLoader(trainLoader, batch_size=batch_size, shuffle=True, num_workers=4), 'val': DataLoader(valLoader, batch_size=batch_size, shuffle=True, num_workers=4) }


	return dataLoader
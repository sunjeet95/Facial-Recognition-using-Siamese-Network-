# Jai Shri Ram | Om Hum Hanumate Namah
# 6th November, 2021
# Sunjeet Jena
# This is the trainer.py file for training the model
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
import torch.optim as optim
import copy
#======================================================================================

def trainer(model, dataLoader, optimizer, criterion, num_epoch=1, threshold_d = 0.1, device = torch.device("cpu")):
	"""
		This function is for training the model for required number of epochs
	"""

	# List to store validation accuracy over each epoch

	lossStats={'train':[],'val':[]}

	bestModel = copy.deepcopy(model)

	maxLoss = 100000

	for epoch in range(num_epoch):

		# Do both training and inference
		print("Epoch: " + str(epoch)+'/'+str(num_epoch-1))

		# Varibable accumulate training loss and validation lostt
		sumLoss = 0

		for phase in dataLoader.keys():

			# Iterate over either training samples or inference samples

			# Check the run-type and set the required model run type
			if(phase =='train'):
				model.train()
			else:
				model.eval()

			steps=0

			# Variable to count the number of correct predictions
			sumCorrectPredictions = 0

			# Variable to count the number of samples
			samples = 0

			for images1, images2, labels in dataLoader[phase]:
				# Set the optimizer gradient to zero

				optimizer.zero_grad()

				# Set the inputs and the labels to device

				inputs1 = images1.to(device)
				inputs2 = images2.to(device)
				labels 	= labels.to(device)

				with torch.set_grad_enabled(phase == 'train'):

					output1 = model(inputs1)
					output2 = model(inputs2)

					loss, dist = criterion(output1, output2, labels)

					if(phase == 'train'):
						loss.backward()
						optimizer.step()


					sumLoss+=loss.item()

				samples+=labels.size()[-1]
				steps+=1
				if(samples%(labels.size()[-1]*100)==0 and phase=='train'):
					print(str(samples))
					print("Running Average for " + phase + " loss: " + str(sumLoss/steps))

			print("\nEpoch Average for " + phase + " loss: " + str(sumLoss/steps))
			
			if(phase=='val' and sumLoss/steps<maxLoss):
				bestModel = copy.deepcopy(model)

			lossStats[phase].append(sumLoss/steps)
		print("-"*40)

	return model, lossStats



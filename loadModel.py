# Jai Shri Ram | Om Hum Hanumate Namah
# 20th November, 2021
# Sunjeet Jena
# This is the loadModel.py file for loading the trained model

#=================Importing the required libraries=====================================
import numpy as np
from PIL import Image
import torch
import math
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import json
import pandas
import torch.optim as optim
# =====================Importing other python files ===================================
import initialize
import dataloader
import trainer
import loss
#======================Reading the Arguments File=======================================
# Open the arguments file
argumentsFile = open("arguments.json" ,'r')
# Parse the json file
arguments = json.load(argumentsFile)
# Read the parameters
learningRate = float(arguments['learningRate'])
batchSize = int(arguments['batchSize'])
num_epochs = arguments['num_epochs']
trainingSamplesFile = arguments['trainingSamplesFile']
validationSamplesFile = arguments['validationSamplesFile']
bestWeightsFile = arguments['bestWeights']
outputNodes = arguments['outputVector']
rootDirectoryImages = arguments['dataRootDirectory']
threshold_step = float(arguments['threshold_step'])
# Close the arguments file
argumentsFile.close()
#========================================================================================



def getModel():

	"""
		This function returns the trained model
	"""

	# Initialize the model
	input_shape, model = initialize.initializeModel(outputNodes = outputNodes)

	# Load the Trained Model Weights
	model.load_state_dict(torch.load(bestWeightsFile))

	return model, input_shape

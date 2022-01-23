# Jai Shri Ram | Om Hum Hanumate Namah
# 5th November, 2021
# Sunjeet Jena
# This is the initialize.py file for initializing the model 

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
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

#======================================================================================

# Define the function to get the 
def initializeModel(featureExtract=True, outputNodes=512):

	# Defining the input size
	input_size = 224

	# Get the resnet101 model
	resnet101 = models.resnet101(pretrained=True)

	# Number of input features
	n_features = resnet101.fc.in_features

	# Update the last layer to ouput only a vector encoding
	resnet101.fc = nn.Sequential(nn.Linear(n_features, outputNodes), nn.ReLU())

	return (input_size, resnet101)
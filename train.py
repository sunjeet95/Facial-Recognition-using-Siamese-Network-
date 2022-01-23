# Jai Shri Ram | Om Hum Hanumate Namah
# 4th November, 2021
# Sunjeet Jena
# This is the train.py file for training the Facial Recognition Neural Network for CSE598:IntrotoDeepLearning Fall2021 Class

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

# Initialize the GPU device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
input_shape, model = initialize.initializeModel(outputNodes = outputNodes)

model.to(device)

# Read the training.csv file
trainSamples = pandas.read_csv(trainingSamplesFile)

# Read the validation.csv file
valSamples = pandas.read_csv(validationSamplesFile)

print("Learning Rate: " + str(learningRate))
print("Batch Size: " + str(batchSize))
print("Training Samples: " + str(trainSamples.shape[0]))
print("Validation Samples: " + str(valSamples.shape[0]))
print("Number of Epochs to be trained: " + str(num_epochs))
print("Threshold Step Size To: " + str(threshold_step))

# Get the dataloader 
dataLoader = dataloader.dataLoader(trainSamples, valSamples, rootDirectoryImages, batchSize, input_shape)

# Initilize the Optimizer
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# Create the loss critertion class
criterion = loss.ContrastiveLoss(margin = 1.0)

print("\n")

# Train the model!!!
bestModel, lossStats = trainer.trainer(model, dataLoader, optimizer, criterion,bestWeightsFile, num_epochs, device) 


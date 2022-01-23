# Jai Shri Ram | Om Hum Hanumate Namah
# 20th November, 2021
# Sunjeet Jena
# This is the findingThreshold.py file for finding the threshold 'd;

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
from tqdm import tqdm
import pickle
# =====================Importing other python files ===================================
import initialize
import dataloader
import trainer
import loss
import loadModel
#======================Reading the Arguments File=======================================
# Open the arguments file
argumentsFile = open("/content/gdrive/MyDrive/myCodes/facialRecognition/arguments.json" ,'r')
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


def findingThreshold():

	# Initialize the GPU device 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model, input_shape = loadModel.getModel()

	model.to(device)

	# Set the model to eval set
	model.eval()

	# Read the training.csv file
	trainSamples = pandas.read_csv(trainingSamplesFile)

	# Read the validation.csv file
	valSamples = pandas.read_csv(validationSamplesFile)

	# Get the dataloader 
	dataLoader = dataloader.dataLoader(trainSamples, valSamples, rootDirectoryImages, batchSize, input_shape)

	# The Loss Function

	# Create the loss critertion class
	criterion = loss.ContrastiveLoss(margin = 1.0)



	best_d = 0.0
	
	best_d_accuracy = 0.0

	a = np.full((int(outputNodes), ), 1)
	b = np.zeros((int(outputNodes)))

	current_d = threshold_step

	max_d = np.linalg.norm(a-b)

	# # We shall only use the validation set
	# for images1, images2, labels in dataLoader['val']:

	d_array = [] 

	while(current_d<max_d):

		predictedTruePositives = 0
		predictedTrueNegatives = 0

		truePositives = 0
		trueNegatives = 0


		for images1, images2, labels in tqdm(dataLoader['val']):

			inputs1 = images1.to(device)
			inputs2 = images2.to(device)
			labels 	= labels.to(device)

			output1 = model(inputs1)
			output2 = model(inputs2)
			_ , dist = criterion(output1, output2, labels)

			similarities = dist > current_d
			
			for predict, label in zip(similarities, labels):

				if(label==0):
					truePositives +=1

					if(predict==label):

						predictedTruePositives += 1

				else:
					trueNegatives +=1

					if(predict==label):

						predictedTrueNegatives += 1


		accuracy = (1/2)*(predictedTruePositives/truePositives + predictedTrueNegatives/trueNegatives)

		print("Accuracy: " + str(accuracy) + " d_value: " + str(current_d))
		
		d_array.append((accuracy, current_d))
		
		if(accuracy>best_d_accuracy):

			best_d_accuracy = accuracy

			best_d = current_d

		current_d += threshold_step

	print("The best Accuracy is: " + str(best_d_accuracy) + " with value_d as: " + str(best_d_accuracy)) 
	
	with open('d_Array.pkl', 'wb') as f:
		pickle.dump(d_array, f)
if __name__=="__main__":

	findingThreshold()
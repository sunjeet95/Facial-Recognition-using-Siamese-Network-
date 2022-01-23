# Jai Shri Ram | Om Hum Hanumate Namah
# 24th November, 2021
# Sunjeet Jena
# This is the inference.py file.

#=================Importing the required libraries=====================================
import numpy as np
from PIL import Image
import argparse
import torch
import math
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import json
import pandas
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
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
outputDirectory = arguments['outputFolder']
# Close the arguments file
argumentsFile.close()
#========================================================================================

def runInference():

	# Argument parser to get the d_threshold
	parser = argparse.ArgumentParser(description='d_threshold')
	parser.add_argument('--d',  type = float, default = 0.05,  help='d_threshold to differentiate between similar or disimilar pairs')
	parser.add_argument('--imageOut',  type = bool, default = False,  help='Generate Output Pair Images')

	args = parser.parse_args()

	# Get the threshold value
	d_threshold = args.d
	imageOut = args.imageOut


	print("The Threshold as been set to: " + str(d_threshold))

	# Create four folders
	# True Positives folder
	if(not os.path.isdir(outputDirectory+'/'+"truePositives")):
		os.makedirs(outputDirectory +'/'+"truePositives")

	# True Negatives Folder
	if(not os.path.isdir(outputDirectory+'/'+"trueNegatives")):
		os.makedirs(outputDirectory +'/'+"trueNegatives")

	# False Positives folder
	if(not os.path.isdir(outputDirectory+'/'+"falsePositives")):
		os.makedirs(outputDirectory +'/'+"falsePositives")

	# True Negatives Folder
	if(not os.path.isdir(outputDirectory+'/'+"falseNegatives")):
		os.makedirs(outputDirectory +'/'+"falseNegatives")


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

	# The Loss Function
	# Create the loss critertion class
	criterion = loss.ContrastiveLoss(margin = 1.0)
	

	# Creating the transform function
	transformImg = transforms.Compose([transforms.Resize((input_shape, input_shape)),
                              	  	transforms.ToTensor(),
                              	  	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	################ Metrics ###############################
	predictedLabels = []
	trueLabels = []
	########################################################

	for index, row in tqdm(valSamples.iterrows()):
		
		# Read the pair of images and their corresponding label
		img1 	= row['pathImg1']
		img2 	= row['pathImg2']
		label 	= row['Result']

		img_1 = os.path.join(rootDirectoryImages,img1)
		img_2 = os.path.join(rootDirectoryImages, img2)

		# Open both the Images
		image1 = Image.open(img_1)
		image2 = Image.open(img_2)

		# # Create a canvas to join the images
		# join_Images = Image.new('RGB',(2*image1.size[0], image1.size[1]), (250,250,250))

		# # Paste the first  Image
		# join_Images.paste(image1,(0,0))
		
		# # Paste the second Image
		# join_Images.paste(image2,(image1.size[0],0))
		

		# ============= Model Input-Ouput ===========================

		image1 = transformImg(image1)
		image2 = transformImg(image2)

		input1 = image1.to(device).unsqueeze(0)
		input2 = image2.to(device).unsqueeze(0)

		output1 = model(input1)
		output2 = model(input2)
		_ , dist = criterion(output1, output2, torch.tensor(label))

		predictLabel = dist > d_threshold
		# ============================================================

		# ============= Prediction Analysis ==========================
		predictedLabels.append(int(predictLabel.item()))
		trueLabels.append(label)



		# If the user argument specifies to generate this image pairs and its predicted output.
		if(imageOut):
			# Similar Pair
			if(label==0):

				# True Positive
				if(predictLabel.item()==label):

					indexPos = len(os.listdir(outputDirectory +'/'+"truePositives"))
					join_Images.save(outputDirectory +'/'+"truePositives"+'/'+str(indexPos) + '.png')


				# False Negative 
				else:

					indexPos = len(os.listdir(outputDirectory +'/'+"falseNegatives"))
					join_Images.save(outputDirectory +'/'+"falseNegatives"+'/'+str(indexPos) + '.png')


			# Dissimilar Pair
			else:

				# True Negative
				if(predictLabel.item()==label):
					indexPos = len(os.listdir(outputDirectory +'/'+"trueNegatives"))
					join_Images.save(outputDirectory +'/'+"trueNegatives"+'/'+str(indexPos) + '.png')

				# False Positive
				else:
					indexPos = len(os.listdir(outputDirectory +'/'+"falsePositives"))
					join_Images.save(outputDirectory +'/'+"falsePositives"+'/'+str(indexPos) + '.png')
			#===========================================================

	with open(os.path.join( outputDirectory, 'predictedLabels.pkl'), 'wb') as f:
		pickle.dump(predictedLabels, f)

	with open(os.path.join( outputDirectory, 'trueLabels.pkl'), 'wb') as f:
		pickle.dump(trueLabels, f)

if __name__=="__main__":

	runInference()
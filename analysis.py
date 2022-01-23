# Jai Shri Ram | Om Hum Hanumate Namah
# 2nd December, 2021
# Sunjeet Jena
# This is the analysis.py file for generating the confusion matrix over the inference results on the validation set and generate the classification report

#=================Importing the required libraries=====================================
import numpy as np
from PIL import Image
import torch
import math
import matplotlib.pyplot as plt
import json
import pandas
import pickle
from sklearn.metrics import confusion_matrix, classification_report

import itertools
#======================================================================================

# Files for the predicted labels

predictedLabelsFile = "output/predictedLabels.pkl"
groundTruthLabelsFile = "output/trueLabels.pkl"


# Load the files

with open(predictedLabelsFile, 'rb') as f:
	predictedLabels = pickle.load(f)

with open(groundTruthLabelsFile, 'rb') as f:
	trueLabels = pickle.load( f)



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

confusion_mtx = confusion_matrix(trueLabels, predictedLabels)
	
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = ["Match", "No Match"])


# Generate Classification Report

report = classification_report(trueLabels, predictedLabels, target_names=['match (0)', 'no match (1)'])

print(report)
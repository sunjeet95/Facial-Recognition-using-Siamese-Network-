"""
Jai Shri Ram | Om Hum Hanumate Namah
6th November, 2022
Sunjeet Jena
"""
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
#======================================================================================


class ContrastiveLoss(torch.nn.Module):
	"""
		This class is a direct implementation from the Medium Article:
		https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec

	"""

	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""

	def __init__(self, margin=1.0, alpha=1.0, beta=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin
		self.alpha = alpha
		self.beta = beta
		self.pdist = nn.PairwiseDistance(p=2, keepdim=False)

	def forward(self, vector1, vector2, label):

		dist = self.pdist(vector1, vector2)

		loss = torch.mean(self.alpha*(1/2*((1-label)* (torch.pow(dist, 2)))) + self.beta*(1/2*((label)*torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))))
		return loss,dist



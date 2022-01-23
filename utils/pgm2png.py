# Jai Shri Ram | Om Hum Hanumate Namah
"""

	This code converts the .PGM image to .PNG image format of the Yale Extended dataset

"""

from PIL import Image
import os

# =========== Input Arguments =========================

inputDirectory = "../dataset/ExtendedYaleB_original"
outputDirectory = "../dataset/extendedPNG_resized"


if(not os.path.isdir(outputDirectory)):
	os.makedirs(outputDirectory)


allFolders = os.listdir(inputDirectory)


for eachFolder in allFolders:

	if(not os.path.isdir(outputDirectory+'/'+eachFolder)):
		os.makedirs(outputDirectory +'/'+eachFolder)

	allFiles =  os.listdir(inputDirectory+'/'+eachFolder)
	for eachImage in allFiles:
		if(eachImage.split('.')[-1]=='pgm'):
			
			img = (Image.open(inputDirectory+'/'+ eachFolder +'/'+eachImage)).convert('RGB')
			img = img.resize((256,256))
			newImageName = eachImage.split('.')
			newImageName.pop()
			newImageName = '.'.join(newImageName)
			img.save(outputDirectory+'/'+eachFolder+'/'+newImageName+'.png', format='png')
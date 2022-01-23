from PIL import Image
from autocrop import Cropper
import os
cropper = Cropper(face_percent=75)

inputDirectory = "../dataset/ExtendedYaleB_PNG_Original"
outputDirectory = "../dataset/croppedFaces"

if(not os.path.isdir(outputDirectory)):
    os.makedirs(outputDirectory)

allFolders = os.listdir(inputDirectory)

for eachFolder in allFolders:

    if(not os.path.isdir(outputDirectory+'/'+eachFolder)):
        os.makedirs(outputDirectory +'/'+eachFolder)

    allFiles =  os.listdir(inputDirectory+'/'+eachFolder)
    
    for eachImage in allFiles:

        img = cropper.crop(inputDirectory+'/'+ eachFolder +'/'+eachImage)

        if img is not None:

            img = Image.fromarray(img)

            img = img.resize((256,256))

            newImageName = eachImage.split('.')
            newImageName.pop()
            newImageName = '.'.join(newImageName)
            img.save(outputDirectory+'/'+eachFolder+'/'+newImageName+'.png', format='png')

        else:

            print("No Face Found!")
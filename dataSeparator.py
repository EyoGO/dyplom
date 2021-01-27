import os
import random
import shutil


# os.mkdir("dishesData")
# os.mkdir("dishesData/train")
# os.mkdir("dishesData/val")

mainDir = "images"
for dirName in os.listdir(mainDir):
#     os.mkdir("dishesData/train/" + dirName)
#     os.mkdir("dishesData/val/" + dirName)
    imageNames = os.listdir(mainDir + "/" + dirName)    
    random.shuffle(imageNames)
    
    trainLen = int(len(imageNames) * 0.8)
    for i in range(0, trainLen):
        shutil.copy(mainDir + "/" + dirName + "/" + imageNames[i], "dishesData/train/" + dirName + "/" + imageNames[i])
    for i in range(trainLen, len(imageNames)):
        shutil.copy(mainDir + "/" + dirName + "/" + imageNames[i], "dishesData/val/" + dirName + "/" + imageNames[i])
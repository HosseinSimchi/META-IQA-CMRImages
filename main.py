#Import libraries 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Models.PreProcessing import PreProcessing
from Models.HyperParametrs import Prior_HyperParameters
from Models.ModelStructures import Model, Classifier

################################ SET GPU OR CPU #################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################ PREPROCESSING #################################

preProcessing = PreProcessing(Image=[], ImageLabel=[]) # We should pass Image data as parameters

[ImageData, ImageLabel] = preProcessing.preProcessing()

[BATCH_SIZE, EPOCHS_PRIOR, LEARNING_RATE, NUM_CLASSES,  KEEP_PROPABILITY, INPUTE_SIZE] = Prior_HyperParameters()

def SPLIT_POINTS(ImageData:list): # We have used Split point to split train and validation datas. (it has been used to update weights twice in every iteration)
    VALIDATION_SPLIT_POINTS = int(len(ImageData)/2)
    return VALIDATION_SPLIT_POINTS

VALIDATION_SPLIT_POINTS = SPLIT_POINTS(ImageData) 
TRAINING_START_POINTS = 0
VALIDATION_START_POINTS = VALIDATION_SPLIT_POINTS

############################# CREATE PRIOR MODEL ######################################3

feature_Extraction = models.resnet18(pretrained=True)
classifier = Classifier(num_classes = NUM_CLASSES , keep_probability = KEEP_PROPABILITY, inputsize = INPUTE_SIZE).to(DEVICE)

PRIOR_MODEL = Model(Feature_Extraction=feature_Extraction, Classifier=classifier).to(DEVICE)

##################### SET OPTIMIZER AND LOSS FUNCTION #######################
OPTIMIZER = optim.Adam(PRIOR_MODEL.parameters(), lr=LEARNING_RATE)
CRITERION = nn.CrossEntropyLoss()
##################### Training Phase ( Prior Meta-Learning Model ) #######################

for epoch in range(EPOCHS_PRIOR):
    pass

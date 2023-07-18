#Import libraries 
import torch
import torch.nn as nn
import numpy as np
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

def SPLIT_POINT(ImageData:list): # We have used Split point to split train and validation datas. (it has been used to update weights twice in every iteration)
    VALIDATION_SPLIT_POINT = int(len(ImageData)/2)
    return VALIDATION_SPLIT_POINT

VALIDATION_SPLIT_POINT = SPLIT_POINT(ImageData) 


############################# CREATE PRIOR MODEL ######################################

feature_Extraction = models.resnet18(pretrained=True)
classifier = Classifier(num_classes = NUM_CLASSES , keep_probability = KEEP_PROPABILITY, inputsize = INPUTE_SIZE).to(DEVICE)

PRIOR_MODEL = Model(Feature_Extraction=feature_Extraction, Classifier=classifier).to(DEVICE)

##################### SET OPTIMIZER AND LOSS FUNCTION #######################
OPTIMIZER = optim.Adam(PRIOR_MODEL.parameters(), lr=LEARNING_RATE)
CRITERION = nn.CrossEntropyLoss()
##################### Training Phase ( Prior Meta-Learning Model ) #######################

for epoch in range(EPOCHS_PRIOR):
    TRAINING_START_POINT = 0
    VALIDATION_START_ = VALIDATION_SPLIT_POINT
    
    SUM_LOSSES = [] # it has been used to sum the losses after two optimisation steps
    
    while ((TRAINING_START_POINT + BATCH_SIZE) <= VALIDATION_START_):
        
        #Split input data into support and query sets.
        SupportSet = ImageData[TRAINING_START_POINT:TRAINING_START_POINT + BATCH_SIZE].to(DEVICE)
        SupportSetLabel = ImageLabel[TRAINING_START_POINT:TRAINING_START_POINT + BATCH_SIZE].to(DEVICE)
        QuerySet = ImageData[VALIDATION_START_:VALIDATION_START_ + BATCH_SIZE].to(DEVICE)
        QuerySetLabel = ImageLabel[VALIDATION_START_:VALIDATION_START_ + BATCH_SIZE ].to(DEVICE)

   

        # The First Level Optimization (Support Set)
        OPTIMIZER.zero_grad()
        OUTPUT_SupportSet = PRIOR_MODEL(SupportSet)
        LOSS_SupportSet = CRITERION(OUTPUT_SupportSet, SupportSetLabel) # To Calculate loss
        SUM_LOSSES.append(LOSS_SupportSet.item())
        LOSS_SupportSet.backward()  
        OPTIMIZER.step() #To Update the weights
        

        # The Second Level Optimization (Query Set)
        OPTIMIZER.zero_grad()
        OUTPUT_QuerySet = PRIOR_MODEL(QuerySet)
        LOSS_QuerySet = CRITERION(OUTPUT_QuerySet, QuerySetLabel)
        SUM_LOSSES.append(LOSS_QuerySet.item())
        LOSS_QuerySet.backward()
        OPTIMIZER.step()

        TRAINING_START_POINT += BATCH_SIZE
        VALIDATION_START_ += BATCH_SIZE

    print("Epoch[{0}/{1}], PriorModel_Loss:{2}".format(epoch + 1, EPOCHS_PRIOR, np.sum(SUM_LOSSES)))

torch.save(PRIOR_MODEL.cuda(),'./PriorModel.pt') #To sAVE prior model to transfer knowledge

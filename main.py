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
    VALIDATION_SPLIT_POINT = int(len(ImageData)/2) #In this article, we have used half of the data for training and the other for validation (you can change it!)
    return VALIDATION_SPLIT_POINT

Validation_Split_Point = SPLIT_POINT(ImageData) 


############################# CREATE PRIOR MODEL ######################################

feature_Extraction = models.resnet18(pretrained=True)
classifier = Classifier(num_classes = NUM_CLASSES , keep_probability = KEEP_PROPABILITY, inputsize = INPUTE_SIZE).to(DEVICE)

Prior_Model = Model(Feature_Extraction=feature_Extraction, Classifier=classifier).to(DEVICE)

##################### SET OPTIMIZER AND LOSS FUNCTION #######################
Optimizer = optim.Adam(Prior_Model.parameters(), lr=LEARNING_RATE)
Criterion = nn.CrossEntropyLoss()
##################### Training Phase ( Prior Meta-Learning Model ) #######################

for epoch in range(EPOCHS_PRIOR):
    
    Training_Start_Point = 0
    Training_End_Point = Training_Start_Point + BATCH_SIZE
    
    Validation_Start_Point = Validation_Split_Point
    Validation_End_Point = Validation_Start_Point + BATCH_SIZE
    
    Sum_Losses = [] # it has been used to sum the losses after two optimisation steps
    
    while (Training_End_Point <= Validation_Start_Point):
        
        #Split input data into support and query sets.
        SupportSet = ImageData[Training_Start_Point:Training_End_Point].to(DEVICE)
        SupportSetLabel = ImageLabel[Training_Start_Point:Training_End_Point].to(DEVICE)
        
        QuerySet = ImageData[Validation_Start_Point:Validation_End_Point].to(DEVICE)
        QuerySetLabel = ImageLabel[Validation_Start_Point:Validation_End_Point].to(DEVICE)

   

        # The First Level Optimization (Support Set)
        Optimizer.zero_grad()
        OUTPUT_SupportSet = Prior_Model(SupportSet)
        LOSS_SupportSet = Criterion(OUTPUT_SupportSet, SupportSetLabel) # To Calculate loss
        Sum_Losses.append(LOSS_SupportSet.item())
        LOSS_SupportSet.backward()  
        Optimizer.step() #To Update the weights
        

        # The Second Level Optimization (Query Set)
        Optimizer.zero_grad()
        OUTPUT_QuerySet = Prior_Model(QuerySet)
        LOSS_QuerySet = Criterion(OUTPUT_QuerySet, QuerySetLabel)
        Sum_Losses.append(LOSS_QuerySet.item())
        LOSS_QuerySet.backward()
        Optimizer.step()

        Training_Start_Point += BATCH_SIZE
        Validation_Start_Point += BATCH_SIZE

    print("Epoch[{0}/{1}], PriorModel_Loss:{2}".format(epoch + 1, EPOCHS_PRIOR, np.sum(Sum_Losses)))

torch.save(Prior_Model.cuda(),'./PriorModel.pt') #To sAVE prior model to transfer knowledge

# Import Libraries 
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Models.PreProcessing import PreProcessing
from Models.HyperParametrs import FineTune_HyperParameters
################################ SET GPU OR CPU #################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################ PREPROCESSING #################################

preProcessing = PreProcessing(Image=[], ImageLabel=[]) # We should pass Image data as parameters

[ImageData, ImageLabel] = preProcessing.preProcessing()

[TERMINATION_CONDITION_COUNTER, MODE1_64_TwoClasses, MODE2_128_TwoClasses, MODE3_256_TwoClasses, MODE1_64_ThreeClasses, MODE2_128_ThreeClasses, MODE3_256_ThreeClasses, EPOCHS_FINETUNE, TERMINATION_CONDITIONS, LEARNING_RATE] = FineTune_HyperParameters()

############################# CREATE PRIOR MODEL ######################################

#Now, we should decide which Modes we want to use it to Train the Model, For example:

for data in MODE1_64_ThreeClasses:
    
    SAVE_ERROR_EARLYSTOPPING = []
    # To Calculate Metrics
    Accuracy, Precision, Recall, f1Score = [], [], [], []
    
    for condition in TERMINATION_CONDITIONS: #we train our model for each condition and calculate the metrics corresponding to each condition, at the final we can find the best condition and the best metrics respectively. ()

        PriorModel = torch.load('./PriorModel.pt')
        OPTIMIZER = optim.Adam(PriorModel.parameters(), lr=LEARNING_RATE)
        CRITERION = nn.CrossEntropyLoss()


# Import Libraries 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Models.PreProcessing import PreProcessing
from Metrics.metrics import Metrics
from Models.HyperParametrs import FineTune_HyperParameters
from Models.ModelStructures import OutPut
################################ SET GPU OR CPU #################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################ PREPROCESSING #################################

preProcessing = PreProcessing(Image=[], ImageLabel=[]) # We should pass Image data as parameters

[ImageData, ImageLabel] = preProcessing.preProcessing()

[Start_Point_traindata,Average_Accuracy_AllFolds, Average_Precision_AllFolds, Average_Recall_AllFolds, Average_F1Score_AllFolds,EPOCHS_FINETUNE, TERMINATION_CONDITIONS, LEARNING_RATE, NUMBER_OF_FOLDS, Fold_Number_Counter] = FineTune_HyperParameters()

################################ ADD METRICS #################################
Accuracy, Precision, Recall = Metrics()
############################# CREATE Fine Tune MODEL ######################################

#Now, we should decide which Modes we want to use it to Train the Model, For example:
BATCH_SIZE = 64 #64, 128 or 256
NUMBER_OF_CLASSES = 2 #It depends on the number of classes that you want to test.
for _ in range(NUMBER_OF_FOLDS):
    
    # To store Metrics per fold
    Acc, Pre, Re, F1 = [], [], [], []
    
    # Getting the end of the training point
    End_Point_traindata = Start_Point_traindata + BATCH_SIZE
    
    for condition in TERMINATION_CONDITIONS: #we train our model for each condition and calculate the metrics corresponding to each condition, at the final we can find the best condition and the best metrics respectively. ()

        Save_Error_Earlystopping = []
        PriorModel = torch.load('./PriorModel.pt') #load the trained model ( Prior Knowledge Model ) 
        Output_FinetuneModel = OutPut(NUMBER_OF_CLASSES).to(DEVICE)
        PARAMS = list(PriorModel.parameters()) + list(Output_FinetuneModel.parameters()) 
        Optimizer = optim.Adam(PARAMS, lr=LEARNING_RATE)
        Criterion = nn.CrossEntropyLoss()
        
        for epoch in range(EPOCHS_FINETUNE):
            
            Image_Finetune_data = ImageData[Start_Point_traindata:End_Point_traindata].to(DEVICE)
            Image_Finetune_Labeldata = ImageLabel[Start_Point_traindata:End_Point_traindata].to(DEVICE)


            Optimizer.zero_grad()
            Model_Outputs = PriorModel(Image_Finetune_data)
            OUTPUT_Predictions = Output_FinetuneModel(Model_Outputs)
            Model_Loss = Criterion(OUTPUT_Predictions, Image_Finetune_Labeldata)
            Model_Loss.backward()
            Optimizer.step()
            
        
            Save_Error_Earlystopping.append(Model_Loss.item())
            
            # We check that if the Loss is less than the condition value, it terminates the training process. (Consider that our Model is taught at least 2 times)
            if len(Save_Error_Earlystopping) > 2:
                if Save_Error_Earlystopping[len(Save_Error_Earlystopping) - 1] < condition:
                    break
                else:
                    continue
            
            
        print("################# Evaluation Part ##################")
        
        PriorModel.eval()
        
        #In this part, we want to manually separate the test data from the training data (as we know, the training and test data will be different each time, and depending on the fold number, the way to separate them will also be different)
        
        
        # If we have given the first fold for training, as a result, it cannot be used for testing, but if we consider other folds, the first fold, which starts from index zero, will definitely be a part of the test data.
        
        Predictions_First_Part, TrueLabel_First_Part = [], [] #setting default values
        Predictions_Second_Part, TrueLabel_Second_Part = [], [] #setting default values
        
        if Fold_Number_Counter != 1: # Except for the first fold
            
            Model_Predicted = PriorModel(Image_Finetune_data[0 : Start_Point_traindata].to(DEVICE))
            OUTPUT_Test_Predictions = Output_FinetuneModel(Model_Predicted)
            predicted = torch.argmax(OUTPUT_Test_Predictions, 1)

            Predictions_First_Part = [predicted[i].item() for i in range(len(predicted))]
            TrueLabel_First_Part = [Image_Finetune_Labeldata[i] for i in range(0 , Start_Point_traindata)]
            
        
        
        if Fold_Number_Counter != NUMBER_OF_FOLDS: # Except for the last fold

            Model_Predicted = PriorModel(Image_Finetune_data[End_Point_traindata:].to(DEVICE))
            OUTPUT_Test_Predictions = Output_FinetuneModel(Model_Predicted)
            predicted = torch.argmax(OUTPUT_Test_Predictions, 1)
            
            Predictions_Second_Part = [predicted[i].item() for i in range(len(predicted))]
            TrueLabel_Second_Part = [Image_Finetune_Labeldata[i] for i in range(End_Point_traindata, len(Image_Finetune_data))]
                
        Target_Value = TrueLabel_First_Part + TrueLabel_Second_Part
        Predicted_Value = Predictions_First_Part + Predictions_Second_Part


        # Obtain the Metrics corresponding to the termination condition
        accuracy = Accuracy(Target_Value, Predicted_Value)
        precision = Precision(Target_Value, Predicted_Value)
        recall = Recall(Target_Value, Predicted_Value)  
        f1Score = (2 * precision * recall) / (precision + recall)
        
        
        Acc.append(accuracy)
        Pre.append(precision)
        Re.append(recall)
        F1.append(f1Score)
    
    # Obtaining the best possible metrics among all termination conditions
    Best_Accuracy = max(Acc)
    index_Best_Score = Acc.index(Best_Accuracy)
    Best_Precision = Pre[index_Best_Score]
    Best_Recall = Re[index_Best_Score]
    Best_F1 = F1[index_Best_Score]
    
    # Adding the best metrics obtained to a separate list to get the average metrics of all folds
    Average_Accuracy_AllFolds.append(Best_Accuracy)
    Average_Precision_AllFolds.append(Best_Precision)
    Average_Recall_AllFolds.append(Best_Recall)
    Average_F1Score_AllFolds.append(Best_F1)
    
    Start_Point_traindata += BATCH_SIZE #Here we go to the next fold.
    Fold_Number_Counter += 1 #it shows the number of fold for cross validation
    


#ACCURACY
print("Average Accuracy is obtained from all tested folds {0}".format((sum(Average_Accuracy_AllFolds)) / NUMBER_OF_FOLDS))
print("The STD of Accuracy is obtained from all tested folds {0}".format(np.std(Average_Accuracy_AllFolds)))

#PRECISION
print("Average Precision is obtained from all tested folds {0}".format((sum(Average_Precision_AllFolds)) / NUMBER_OF_FOLDS))
print("The STD of Precision is obtained from all tested folds {0}".format(np.std(Average_Precision_AllFolds)))

#RECALL
print("Average Recall is obtained from all tested folds {0}".format((sum(Average_Recall_AllFolds)) / NUMBER_OF_FOLDS))
print("The STD of Recall is obtained from all tested folds {0}".format(np.std(Average_Recall_AllFolds)))

#F1-MEASURE
print("Average F1-Score is obtained from all tested folds {0}".format((sum(Average_F1Score_AllFolds)) / NUMBER_OF_FOLDS))
print("The STD of F1-Score is obtained from all tested folds {0}".format(np.std(Average_F1Score_AllFolds)))

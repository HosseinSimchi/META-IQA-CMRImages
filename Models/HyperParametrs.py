def Prior_HyperParameters() :
    BATCH_SIZE = 64
    EPOCHS_PRIOR = 30
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5 
    KEEP_PROPABILITY = 0.5
    INPUTE_SIZE = 1000, #The default output of Resnet18 network
    
    return [BATCH_SIZE, EPOCHS_PRIOR, LEARNING_RATE, NUM_CLASSES,  KEEP_PROPABILITY, INPUTE_SIZE]
    
    
def FineTune_HyperParameters() :
    
    NUMBER_OF_FOLDS = 160;
    Fold_Number_Counter = 1
    Start_Point_traindata = 0
    EPOCHS_FINETUNE = 500;
    

    TERMINATION_CONDITIONS = [10, 8, 6, 4, 3.5, 3, 2.5, 2, 1.5] #These numbers are determined based on the GridSearch algorithm. (As we know, this algorithm initially takes long steps to determine the value of hyperparameters and takes shorter steps over time. For this reason, the distance of 2 is selected at first, and the closer the error value is to we reach zero, we reduce the distance to 0.5)
    
    LEARNING_RATE = 1e-4 
    Average_Accuracy_AllFolds, Average_Precision_AllFolds, Average_Recall_AllFolds, Average_F1Score_AllFolds = [], [], [], []
    
    return [Start_Point_traindata,Average_Accuracy_AllFolds, Average_Precision_AllFolds, Average_Recall_AllFolds, Average_F1Score_AllFolds,EPOCHS_FINETUNE, TERMINATION_CONDITIONS, LEARNING_RATE, NUMBER_OF_FOLDS, Fold_Number_Counter]

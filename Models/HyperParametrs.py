import torch.optim as optim
import torch.nn as nn

def Prior_HyperParameters() :
    BATCH_SIZE = 64
    EPOCHS_PRIOR = 30
    LEARNING_RATE = 1e-4
    OPTIMIZER = optim.Adam()
    CRITERION = nn.CrossEntropyLoss()
    NUM_CLASSES = 5
    KEEP_PROPABILITY = 0.5
    INPUTE_SIZE = 1000
    
    return BATCH_SIZE, EPOCHS_PRIOR, LEARNING_RATE, OPTIMIZER, CRITERION, NUM_CLASSES,  KEEP_PROPABILITY, INPUTE_SIZE
    
    

TERMINATION_CONDITION_COUNTER = 0
SAVE_ERROR_EARLYSTOPPING = []



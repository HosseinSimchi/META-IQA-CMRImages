def Prior_HyperParameters() :
    BATCH_SIZE = 64
    EPOCHS_PRIOR = 30
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2 or 3 #it depends on the number of classes
    KEEP_PROPABILITY = 0.5
    INPUTE_SIZE = 1000,
    
    return [BATCH_SIZE, EPOCHS_PRIOR, LEARNING_RATE, NUM_CLASSES,  KEEP_PROPABILITY, INPUTE_SIZE]
    
    
def FineTune_HyperParameters() :
    TERMINATION_CONDITION_COUNTER = 0
    
    
    # If we have two classes in the FineTune set, then :
    
    MODE1_64_TwoClasses = [x for x in range(0, 10240 , 64)]
    MODE2_128_TwoClasses = [x for x in range(0, 20480 , 128)]
    MODE3_256_TwoClasses = [x for x in range(0, 40960, 256)]
    
    # If we have three classes in the FineTune set, then :
    MODE1_64_ThreeClasses = [x for x in range(0, 10176, 64)]
    MODE2_128_ThreeClasses = [x for x in range(0, 20352, 128)]
    MODE3_256_ThreeClasses = [x for x in range(0, 40192, 256)]
    
    
    EPOCHS_FINETUNE = 500;
    TERMINATION_CONDITIONS = [10, 8, 6, 4, 3.5, 3, 2.5, 2, 1.5]
    LEARNING_RATE = 1e-4 
    
    return [TERMINATION_CONDITION_COUNTER, MODE1_64_TwoClasses, MODE2_128_TwoClasses, MODE3_256_TwoClasses, MODE1_64_ThreeClasses, MODE2_128_ThreeClasses, MODE3_256_ThreeClasses, EPOCHS_FINETUNE, TERMINATION_CONDITIONS, LEARNING_RATE]
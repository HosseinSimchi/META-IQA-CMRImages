#import libraries
import torch
import numpy as np


class PreProcessing:
    
    def __init__ (self,Image:list, ImageLabel:list):
        self.Image = Image
        self.ImageLabel = ImageLabel
    
        
    def preProcessing(self):
        #It depends on your dataset, But we have done this before fit the model:
        self.Image = np.array(self.Image)
        self.Image = self.Image.astype('float32') 
        self.Image = self.Image.reshape((self.Image.shape[0], 1, 90, 90)) #reshaped to (90,90) images
        self.Image = torch.tensor(self.Image)
        self.Image = self.Image.repeat(1, 3, 1, 1) #we have used this for resnet inputs (resnet accepts only three channels)
        
        self.ImageLabel = torch.tensor(self.ImageLabel)
        
        return [self.Image, self.ImageLabel]
    
#Import Libraries
import torch.nn as nn

class ModelStructure(nn.Module):
    def __init__(self , resnetType, net):
        super(ModelStructure, self).__init__()
        self.resnet_layer = resnetType
        self.baselineModel = net


    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.baselineModel(x)

        return x


class BaselineModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.LogSoftmax()

    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)

        return out


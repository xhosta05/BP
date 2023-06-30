import torch
import torch.nn as nn

class Simple1DCNN(torch.nn.Module):
    def __init__(self,classes):
        super(Simple1DCNN, self).__init__()

        # self.bn1 = nn.BatchNorm1d(infeatures)
        
        self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=20, kernel_size=3, stride=2)
        self.act1 = torch.nn.ReLU()

#         self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=3, kernel_size=1)
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.act2 = torch.nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.lin = nn.Linear(256, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.act2(x)

        x = self.avgpool(x)
        x = self.lin(x)
        return x


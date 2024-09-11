import torch
import torch.nn as nn
import hiddenlayer as hl

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_conv_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU())
        self.down_conv_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.down_conv_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())
        self.down_conv_4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU())
        
        self.up_conv_4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU())
        self.up_conv_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU())
        self.up_conv_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU())
        self.up_conv_1 = nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2), nn.ReLU())
        
    def forward(self, x):
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(nn.MaxPool2d(2)(x1))
        x3 = self.down_conv_3(nn.MaxPool2d(2)(x2))
        x4 = self.down_conv_4(nn.MaxPool2d(2)(x3))
        
        x = self.up_conv_4(nn.MaxPool2d(2)(x4))
        x = self.up_conv_3(nn.MaxPool2d(2)(x + x3))
        x = self.up_conv_2(nn.MaxPool2d(2)(x + x2))
        x = self.up_conv_1(nn.MaxPool2d(2)(x + x1))
        
        return x

# Create a model and visualise
model = UNet()
hl.build_graph(model, torch.zeros([1, 1, 224, 224]))




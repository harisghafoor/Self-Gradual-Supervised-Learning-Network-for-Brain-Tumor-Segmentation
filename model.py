import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import weight_norm

class GaussianNoise(nn.Module):
    
    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape))
        self.std = std
        
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise
class convnet(nn.Module):
    
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(convnet, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = GaussianNoise(batch_size, std=self.std)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = convnet(100, 0.05)
    input_x = torch.randn(100, 1, 28, 28)
    output_x = model(input_x)
    print(input_x.shape)
    print(output_x.shape)
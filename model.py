import torch
import torch.nn as nn


def pad2d(input_, target, device = 'cpu'):
    output_ = torch.zeros((target.shape[0],input_.shape[1],target.shape[2],target.shape[3]),device=device)
    start_idx = int((output_.shape[-1] - input_.shape[-1]) / 2)
    try:
        output_[:,:,start_idx:-start_idx,start_idx:-start_idx] = input_
    except:
        try:
            output_[:,:,start_idx:-start_idx-1,start_idx:-start_idx-1] = input_
        except:
            output_[:,:,1:,1:] = input_
    return output_


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        n = 1
        #layers
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 2, out_channels = int(64/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(64/n)),
                               nn.Conv2d(in_channels = int(64/n), out_channels = int(64/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(64/n)))
        
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = int(64/n), out_channels = int(128/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(128/n)),
                               nn.Conv2d(in_channels = int(128/n), out_channels = int(128/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(128/n)))
        
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = int(128/n), out_channels = int(256/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(256/n)),
                               nn.Conv2d(in_channels = int(256/n), out_channels = int(256/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(256/n)))
        
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels = int(256/n), out_channels = int(512/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(512/n)),
                               nn.Conv2d(in_channels = int(512/n), out_channels = int(512/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(512/n)))
        
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels = int(512/n), out_channels = int(1024/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(1024/n)),
                               nn.Conv2d(in_channels = int(1024/n), out_channels = int(1024/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(1024/n)),
                               nn.ConvTranspose2d(in_channels = int(1024/n), out_channels = int(512/n), kernel_size = 2, stride= 2),nn.ReLU(),nn.BatchNorm2d(int(512/n)))
        
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels = int(1024/n), out_channels = int(512/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(512/n)),
                               nn.Conv2d(in_channels = int(512/n), out_channels = int(512/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(512/n)),
                               nn.ConvTranspose2d(in_channels = int(512/n), out_channels = int(256/n), kernel_size = 2, stride= 2),nn.ReLU(),nn.BatchNorm2d(int(256/n)))
        
        self.layer7 = nn.Sequential(nn.Conv2d(in_channels = int(512/n), out_channels = int(256/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(256/n)),
                               nn.Conv2d(in_channels = int(256/n), out_channels = int(256/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(256/n)),
                               nn.ConvTranspose2d(in_channels = int(256/n), out_channels = int(128/n), kernel_size = 2, stride= 2),nn.ReLU(),nn.BatchNorm2d(int(128/n)))
        
        self.layer8 = nn.Sequential(nn.Conv2d(in_channels = int(256/n), out_channels = int(128/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(128/n)),
                               nn.Conv2d(in_channels = int(128/n), out_channels = int(128/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(128/n)),
                               nn.ConvTranspose2d(in_channels = int(128/n), out_channels = int(64/n), kernel_size = 2, stride= 2),nn.ReLU(),nn.BatchNorm2d(int(64/n)))
        
        self.layer9 = nn.Sequential(nn.Conv2d(in_channels = int(128/n), out_channels = int(64/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(64/n)),
                               nn.Conv2d(in_channels = int(64/n), out_channels = int(64/n), kernel_size = 3),nn.ReLU(),nn.BatchNorm2d(int(64/n)))
        
        self.out = nn.Conv2d(in_channels = int(64/n), out_channels = 1, kernel_size = 1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        
    def forward(self,x,device = 'cpu'):
        
        y1 = self.layer1(x)
        y = self.maxpool(y1)
        
        y2 = self.layer2(y)
        y = self.maxpool(y2)
        
        y3 = self.layer3(y)
        y = self.maxpool(y3)
        
        y4 = self.layer4(y)
        y = self.maxpool(y4)
        
        y = self.layer5(y)
        
        y = torch.cat((y4,pad2d(y, y4, device = device)),dim = 1)
        y = self.layer6(y)
        
        y = torch.cat((y3,pad2d(y, y3, device = device)),dim = 1)
        y = self.layer7(y)
        
        y = torch.cat((y2,pad2d(y, y2, device = device)),dim = 1)
        y = self.layer8(y)
        
        y = torch.cat((y1,pad2d(y, y1, device = device)),dim = 1)
        y = self.layer9(y)
        
        y = pad2d(y, x, device = device)
        
        y = self.out(y)
        
        return y
    

def test(device = 'cpu'):    
    a = torch.ones((4,1,140,140),device=device)
    
    model = UNet().to(device)
    print(model)
    
    b = model(a,device)
    
    print(a.shape)
    print(b.shape)
        
        
if __name__ == '__main__':
    test('cpu')

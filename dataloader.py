import torch
from torchvision.transforms import functional as tvf
import pandas as pd
import random
from matplotlib import pyplot as plt
from overlap import generate_data

from numpy import stack


ROOT_PATH = 'PATH/TO/YOUR/FOLDER/'


def rand_augment(xi, yi):
    if random.random() > 0.25:
        rotation = random.randint(0, 360)
        xi = tvf.rotate(xi, angle = rotation)
        yi = tvf.rotate(yi, angle = rotation)
        
    if random.random() > 0.5:
        xi = tvf.hflip(xi)
        yi = tvf.hflip(yi)
        
    if random.random() > 0.5:
        xi = tvf.vflip(xi)
        yi = tvf.vflip(yi)
    
    return xi, yi


def augment_batch(x, y, p=0.5):
    new_x = []
    new_y = []
    
    xshape = x.shape
    yshape = y.shape
    
    xtemp = torch.zeros(xshape, dtype=torch.float)
    ytemp = torch.zeros(yshape, dtype=torch.float)
    
    for i in range(x.shape[0]):
        if random.random() > p:
            x_, y_ = rand_augment(x[i],y[i])
            new_x.append(x_.view(1,xshape[1],xshape[2],xshape[3]))
            new_y.append(y_.view(1,yshape[1],yshape[2],yshape[3]))
            
        else:
            new_x.append(x[i].view(1,xshape[1],xshape[2],xshape[3]))
            new_y.append(y[i].view(1,yshape[1],yshape[2],yshape[3]))
            
    x = torch.cat(new_x,dim=0,out=xtemp)
    y = torch.cat(new_y,dim=0,out=ytemp)

    return x, y


def update_path(root_path = ''):
    global ROOT_PATH
    print('Path changed from %s to'%(ROOT_PATH), end=' ')
    ROOT_PATH = root_path
    print(ROOT_PATH)
    
    
def load_batch_dataset(data):
    x, mask = torch.from_numpy(data[:,0:2]).to(dtype=torch.float), torch.from_numpy(data[:,2]).to(dtype=torch.float)
    mask = mask.view(mask.shape[0],1,mask.shape[1],mask.shape[2])
    return x, mask

    
class dataloader(): #load all the data, convert to torch, randomize
    def __init__(self, batch = 32, data_file = 'MIM_ exports_NEW_extension/',post = False, augment = True):
        csv = pd.read_csv(ROOT_PATH + 'MIM data location - MIM current corrected data .csv')
        lookup = csv.to_numpy()        
        data_list = [[data_file+a[0]+'/'+a[1], data_file+a[0]+'/'+a[3], data_file+a[0]+'/'+a[5], data_file+a[0]+'/'+a[7], data_file+a[0]+'/'+a[6]] for a in lookup]     
        self.data = generate_data(ROOT_PATH, data_list)[:2500]
        
        self.augment = augment
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        
        self.info = {"samples" : len(self.data),
                     "batch_size" : self.batch,
                     "augment" : self.augment}
        
    def randomize(self):
        sample_len = len(self.data)
        self.idx = random.sample(range(0, sample_len), sample_len)
    
    def load_batch(self, post = False):        
        if self.Flag: #only runs the first time 
            self.randomize()
            self.Flag = False
            
        max_id = len(self.data) - 1
        
        if self.id + self.batch > max_id:         
            if self.id < max_id:
                batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:]])
            elif self.id == max_id:
                batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:self.id]])
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:self.id + self.batch]])
            self.id += self.batch
                    
        if self.augment:
            batch_raw, batch_mask = augment_batch(batch_raw, batch_mask, 0.75)
        
        return batch_raw, batch_mask
    
    def data_info(self):
        return self.info
    
    
class dataloader_val(): #load all the data, convert to torch, randomize
    def __init__(self, batch = 32, data_file = 'MIM_ exports_NEW_extension/', post = False):
        csv = pd.read_csv(ROOT_PATH + 'MIM data location - MIM current corrected data .csv')
        lookup = csv.to_numpy()        
        data_list = [[data_file+a[0]+'/'+a[1], data_file+a[0]+'/'+a[3], data_file+a[0]+'/'+a[5], data_file+a[0]+'/'+a[7], data_file+a[0]+'/'+a[6]] for a in lookup]     
        self.data = generate_data(ROOT_PATH, data_list)[2500:]
        
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        
        self.info = {"samples" : len(self.data),
                     "batch_size" : self.batch}
        
    def randomize(self):
        sample_len = len(self.data)
        self.idx = random.sample(range(0, sample_len), sample_len)
    
    def load_batch(self, post = False):        
        if self.Flag: #only runs the first time 
            self.randomize()
            self.Flag = False
            
        max_id = len(self.data) - 1
        
        if self.id + self.batch > max_id:         
            if self.id < max_id:
                batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:]])
            elif self.id == max_id:
                batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:self.id]])
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw, batch_mask = load_batch_dataset(self.data[self.idx[self.id:self.id + self.batch]])
            self.id += self.batch
            
        return batch_raw, batch_mask
    
    def data_info(self):
        return self.info
        
    
def test():
    update_path('E:/ML/Ilka_segmentation/')
    
    print('Training data:')
    dt = dataloader(batch = 32, data_file='MIM_ exports_NEW_extension/',post=True)
    print(dt.data_info())
    for i in range(1000):
        x,y = dt.load_batch()
        if i % 10 == 0:
            print(x.shape,y.shape)
            print(dt.id)
            plt.imshow(stack([x[0,0].detach().cpu().numpy().T,x[0,1].detach().cpu().numpy().T,y[0,0].detach().cpu().numpy().T],2))
            plt.show()
    
    print('Validation data:')
    dt = dataloader_val(batch = 32, data_file='MIM_ exports_NEW_extension/',post=True)
    print(dt.data_info())
    for i in range(1000):
        x,y = dt.load_batch()
        if i % 10 == 0:
            print(x.shape,y.shape)
            print(dt.id)
            plt.imshow(stack([x[0,0].detach().cpu().numpy().T,x[0,1].detach().cpu().numpy().T,y[0,0].detach().cpu().numpy().T],2))
            plt.show()
    
    
if __name__ == '__main__':
    test()

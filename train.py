import sys
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
# from build_model import *
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
#from dataloader import npy_train,npy_valid
from dataloader_un import npy_train,npy_valid
# from model import ResNetUNet
from model import UNet11
import numpy as np
#from Transforms import *
from build_unetmodel import UNet
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#--------------------------
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count
#------------------------------        



#----------------------------------------

# transformations_train = transforms.Compose([transforms.ToTensor()])
    
# transformations_val = transforms.Compose([transforms.ToTensor()])  
   
#--------------- Mean calculation ---------------------------------------------- 

# def mean_calc():
#         epislon = 10E-5
#         x_train = np.load('/home/SSD/protien_ICMLA/XTEST_channels_8_torch.npy')
#         y_train = np.load('/home/SSD/protien_ICMLA/YTEST_channels_8_torch.npy')
#         y_train = np.log(y_train+epislon)
#         x_train = np.log(x_train+epislon)
#         np.save('LOG_XTEST_channels_8_torch',x_train)
#         np.save('LOG_YTEST_channels_8_torch',y_train)
#         # ytrain_mean = y_train.mean()
#         # ytrain_max = y_train.max()
#         # xtrain_mean = x_train.mean()
#         # xtrain_max = x_train.max()
#         # print(np.array([xtrain_mean,xtrain_max,ytrain_mean,ytrain_max]))
#         # a=np.array([xtrain_mean,xtrain_max,ytrain_mean,ytrain_max])
#         # np.save("LOG_mean_max_train",a)
# mean_calc()        
# exit()                                    

class mse(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(mse, self).__init__()
        
    def forward(self, PRED, YTRUE):
        # print('MSE started')
        train_loss1 = F.mse_loss(PRED, YTRUE, True)
        # print('MSE calculated')
        return train_loss1
        
       
class mae(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(mae, self).__init__()
        
    def forward(self, PRED, YTRUE):
        # print('MAE started')
        train_loss1 = F.smooth_l1_loss(PRED, YTRUE, True)
        # print('MAE calculated')
        return train_loss1

class symmetry_mse(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(symmetry_mse, self).__init__()
        
    def forward(self, y_pred):
    
        # print('sym_mse started')
        train_loss1 = F.mse_loss(y_pred, y_pred.permute(0,1,3,2), True)
        # print('sym_mse calculated')
        return train_loss1

#--------------
LOGGING = True
DATA_DIR='/home/SSD/protien_ICMLA/'
BATCH_SIZE = 16  
NUM_EPOCHS = 500
LEARNING_RATE = 10E-4
LEARNING_RATE_DECAY = 0.8
WEIGHT_TENSOR = torch.Tensor([2,2,2,2,2]).cuda()
WEIGHT_DECAY = 0.01
MODEL = 'vgg11'
SUMMARY_NAME = 'PROTIEN'+'_'+str(BATCH_SIZE)+'_'+str(LEARNING_RATE)+'_'+str(LEARNING_RATE_DECAY)+'_'+str(WEIGHT_DECAY)+'_'+str(MODEL)+"_dilation_normalized_UNorm_CEloss"

if LOGGING:
    writer = SummaryWriter('./runs/'+SUMMARY_NAME)
    print("**********LOGGING IS ON***************\n"*10)
else:
    print("**********LOGging IS Off***************\n"*10)
  
def train():
    cuda = torch.cuda.is_available()
    net = UNet11(1,32,8)
    if cuda:
        net = net.cuda()
    #-------------------------------------
    # for warm restart
    #net.load_state_dict(torch.load('path for pth file'))
    #-------------------------
    criterion1 = mse().cuda()
    criterion2 = mae().cuda()
    criterion3 = symmetry_mse().cuda()
    criterion4 = nn.BCELoss().cuda()
    criterion5 = symmetry_mse().cuda()
    #-------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE,weight_decay = WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[5,25,75,125,200], gamma=LEARNING_RATE_DECAY)
    #----------------------------------------------
    print("preparing training data ...")
    train_dataset = npy_train(os.path.join(DATA_DIR),transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
    test_dataset = npy_valid(os.path.join(DATA_DIR),transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)
    #-----------------------------
    train_iter_count = int(len(train_loader.dataset)/BATCH_SIZE)
    test_iter_count = int(len(test_loader.dataset)/BATCH_SIZE)    
    print("done loading data ...")
    # ------ train -------
    for epoch in tqdm(range(NUM_EPOCHS)):
        scheduler.step()        
        epoch_train_loss = Average()
        epoch_mse_loss = Average()
        epoch_mae_loss = Average()
        epoch_sym_loss = Average()        
        epoch_ce_loss = Average()        
        epoch_ce_sym_loss = Average()        

        net.train()
        
        for i, (x_sample, y_sample) in tqdm(enumerate(train_loader)):
            images = Variable(x_sample.float())
            masks = Variable(y_sample.float())
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs, ce_outputs = net(images)

            c1 = criterion1(outputs,masks)  ### mse loss
            c2 = criterion2(outputs,masks)  ## mae loss
            c3 = criterion3(outputs) ## symmeteric losss

            c4 = criterion4(ce_outputs, (masks < 8).float())  ##for BCE loss
            c5 = criterion5(ce_outputs) ##for symmetry in BCE


            #-----------------------------------
            epoch_mse_loss.update(c1.data[0], images.size(0))
            epoch_mae_loss.update(c2.data[0], images.size(0))
            epoch_sym_loss.update(c3.data[0], images.size(0))   
            epoch_ce_loss.update(c4.data[0], images.size(0))   
            epoch_ce_sym_loss.update(c5.data[0], images.size(0))   
            #-----------------losses-------------         
            loss = WEIGHT_TENSOR[0]*c1 + WEIGHT_TENSOR[1]*c2 + WEIGHT_TENSOR[2]*c3 + WEIGHT_TENSOR[3] * c4 + WEIGHT_TENSOR[4] * c5

            
            epoch_train_loss.update(loss.data[0], images.size(0))
            
            #------------logging----------------
            if LOGGING:
                writer.add_scalar('Train Batch Loss',loss.data[0],i+epoch*train_iter_count)
                writer.add_scalar('Train Batch mse Loss',c1.data[0],i+epoch*train_iter_count)
                writer.add_scalar('Train Batch mae  Loss',c2.data[0],i+epoch*train_iter_count)
                writer.add_scalar('Train Batch sym Loss',c3.data[0],i+epoch*train_iter_count)                                    
                writer.add_scalar('Train Batch CE loss Loss',c4.data[0],i+epoch*train_iter_count)                                    
                writer.add_scalar('Train Batch CE sym Loss',c5.data[0],i+epoch*train_iter_count)                                    
            #------backward-----    
            loss.backward()
            optimizer.step()

            if LOGGING:
                for param_group in optimizer.param_groups:
                    writer.add_scalar('Learning Rate',param_group['lr'])

        print("Epoch: {}, Epoch Loss:{}, epoch mse loss:{}, epoch mae loss:{}, epoch sym loss:{}, epoch CE loss:{}, epoch CE sym loss:{}".format(epoch,epoch_train_loss.avg,epoch_mse_loss.avg,epoch_mae_loss.avg,epoch_sym_loss.avg, epoch_ce_loss.avg, epoch_ce_sym_loss.avg))
        
        if LOGGING:
            writer.add_scalar('Train Epoch Loss',epoch_train_loss.avg,epoch)
            writer.add_scalar('Train Epoch mse Loss',epoch_mse_loss.avg,epoch)
            writer.add_scalar('Train Epoch mae Loss',epoch_mae_loss.avg,epoch)
            writer.add_scalar('Train Epoch sym Loss',epoch_sym_loss.avg,epoch)          
            writer.add_scalar('Train Epoch CE Loss',epoch_sym_loss.avg,epoch)          
            writer.add_scalar('Train Epoch CE sym Loss',epoch_sym_loss.avg,epoch)          
        ######################Validation###########
        
        val_loss1 = Average()
        val_loss2 = Average()
        val_loss3 = Average()
        val_loss4 = Average()
        val_loss5 = Average()
        val_loss = Average()
        
        net.eval()
        print("starting validation")
        for i ,(images_val, masks_val) in tqdm(enumerate(test_loader)):

            images_val = Variable(images_val.float())
            masks_val = Variable(masks_val.float())
            if cuda:
                images_val = images_val.cuda()
                masks_val = masks_val.cuda()

            outputs, ce_outputs = net(images_val)
        
            c1 = criterion1(outputs, masks_val)  #mse_loss
            c2 = criterion2(outputs, masks_val)  #mae_loss
            c3 = criterion3(outputs) #symmetric loss
            c4 = criterion4(ce_outputs, (masks_val < 8).float())
            c5 = criterion5(ce_outputs)

            vloss = WEIGHT_TENSOR[0]*c1 + WEIGHT_TENSOR[1]*c2 + WEIGHT_TENSOR[2]*c3 +WEIGHT_TENSOR[3]*c4 + WEIGHT_TENSOR[4]*c5

            val_loss.update(vloss.data[0], images_val.size(0))            
            val_loss1.update(c1.data[0], images_val.size(0))
            val_loss2.update(c2.data[0], images_val.size(0))
            val_loss3.update(c3.data[0], images_val.size(0))                
            val_loss4.update(c4.data[0], images_val.size(0))                
            val_loss5.update(c5.data[0], images_val.size(0))                
               
            if LOGGING:
                writer.add_scalar('val Batch Loss',vloss.data[0],i+epoch*test_iter_count)
                writer.add_scalar('val Batch mse Loss',c1.data[0],i+epoch*test_iter_count)
                writer.add_scalar('val Batch mae Loss',c2.data[0],i+epoch*test_iter_count)
                writer.add_scalar('val Batch sym Loss',c3.data[0],i+epoch*test_iter_count) 
                writer.add_scalar('val Batch CE Loss',c3.data[0],i+epoch*test_iter_count) 
                writer.add_scalar('val Batch CE sym Loss',c3.data[0],i+epoch*test_iter_count) 

        print("Epoch: {}, Epoch VLoss:{}, epoch mse Vloss:{}, epoch mae Vloss:{}, epoch sym Vloss:{}, epoch CE Vloss:{},epoch CE sym Vloss:{}".format(epoch,val_loss.avg,val_loss1.avg,val_loss2.avg,val_loss3.avg,val_loss4.avg,val_loss5.avg))

        if LOGGING:
            writer.add_scalar('val Epoch Loss',val_loss.avg,epoch)
            writer.add_scalar('val Epoch mse Loss',val_loss1.avg,epoch)
            writer.add_scalar('val Epoch mae Loss',val_loss2.avg,epoch)
            writer.add_scalar('val Epoch sym Loss',val_loss3.avg,epoch)
            writer.add_scalar('val Epoch CE Loss',val_loss3.avg,epoch)
            writer.add_scalar('val Epoch CE sym Loss',val_loss3.avg,epoch)

        torch.save(net.state_dict(),SUMMARY_NAME+'_'+str(epoch)+'.pth')
    return net

if __name__ == "__main__":
    print("I am in train")
    train()
